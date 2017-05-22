// Copyright 2017 Ben Frederickson
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nmslib/similarity_search/include/init.h"
#include "nmslib/similarity_search/include/index.h"
#include "nmslib/similarity_search/include/space.h"
#include "nmslib/similarity_search/include/knnquery.h"
#include "nmslib/similarity_search/include/knnqueue.h"
#include "nmslib/similarity_search/include/methodfactory.h"
#include "nmslib/similarity_search/include/spacefactory.h"

namespace py = pybind11;

namespace similarity {
const char * module_name = "nmsbind";

enum DistType {
    DISTTYPE_FLOAT,
    DISTTYPE_DOUBLE,
    DISTTYPE_INT
};

enum DataType {
    DATATYPE_DENSE_VECTOR,
    DATATYPE_SPARSE_VECTOR,
    DATATYPE_OBJECT_AS_STRING,
};

// forward references
template <typename dist_t> void exportIndex(py::module * m);
template <typename dist_t> ObjectVector loadObjectVector(py::object data);
template <typename dist_t> std::string distName();
AnyParams loadParams(py::object o);
DistType distTypeFromObject(py::object data);

// Wrap a space/objectvector/index together for ease of use
template <typename dist_t>
struct IndexWrapper {
    IndexWrapper(const std::string & method,
                 const std::string & space_type,
                 py::object data_,
                 py::object space_params)
            : method(method), space_type(space_type),
              space(SpaceFactoryRegistry<dist_t>::Instance().CreateSpace(space_type,
                        loadParams(space_params))),
              data(loadObjectVector<dist_t>(data_)) {
    }

    void createIndex(py::object index_params) {
        auto factory = MethodFactoryRegistry<dist_t>::Instance();
        index.reset(factory.CreateMethod(false, method, space_type, *space, data));

        AnyParams params = loadParams(index_params);

        py::gil_scoped_release l;
        index->CreateIndex(params);
    }

    void loadIndex(const std::string & filename) {
        auto factory = MethodFactoryRegistry<dist_t>::Instance();
        index.reset(factory.CreateMethod(false, method, space_type, *space, data));

        py::gil_scoped_release l;
        index->LoadIndex(filename);
    }

    void saveIndex(const std::string & filename) {
        if (!index) {
            throw std::invalid_argument("Must call createIndex or loadIndex before this method");
        }
        py::gil_scoped_release l;
        index->SaveIndex(filename);
    }

    py::object knnQuery(py::array_t<dist_t> v, int k) {
        if (!index) {
            throw std::invalid_argument("Must call createIndex or loadIndex before  this method");
        }
        Object obj(0, -1, v.size() * sizeof(dist_t), v.data(0));
        KNNQuery<dist_t> knn(*space, &obj, k);
        {
            py::gil_scoped_release l;
            index->Search(&knn, -1);
        }

        // Create numpy arrays for the output
        size_t size = knn.Result()->Size();
        py::array_t<int> ids(size);
        py::array_t<dist_t> distances(size);
        auto raw_ids = ids.mutable_unchecked();
        auto raw_distances = distances.mutable_unchecked();

        // Iterate over the queue, getting a copy so we can
        // destructively pop elements off of it
        std::unique_ptr<KNNQueue<dist_t>> res(knn.Result()->Clone());
        while (!res->Empty() && size > 0) {
            // iterating here in reversed order, undo that
            size -= 1;
            raw_ids(size) = res->TopObject()->id();
            raw_distances(size) = res->TopDistance();
            res->Pop();
        }

        return py::make_tuple(ids, distances);
    }

    std::string repr() const {
        std::stringstream ret;
        ret << "<" << module_name << "." << distName<dist_t>() << "Index method='" << method
            << "' space='" << space_type <<  "' at " << this << ">";
        return ret.str();
    }

    ~IndexWrapper() {
        for (auto datum : data) {
            delete datum;
        }
    }

    std::string method;
    std::string space_type;
    std::unique_ptr<Space<dist_t>> space;
    std::unique_ptr<Index<dist_t>> index;
    ObjectVector data;
};


PYBIND11_PLUGIN(nmsbind) {
    // TODO(ben): set up logging
    initLibrary();
    py::module m(module_name, "Bindings for Non-Metric Space Library (NMSLIB)");

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    py::enum_<DistType>(m, "DistType")
        .value("FLOAT", DISTTYPE_FLOAT)
        .value("DOUBLE", DISTTYPE_DOUBLE)
        .value("INT", DISTTYPE_INT);

    py::enum_<DataType>(m, "DataType")
        .value("DENSE_VECTOR", DATATYPE_DENSE_VECTOR)
        .value("SPARSE_VECTOR", DATATYPE_SPARSE_VECTOR)
        .value("OBJECT_AS_STRING", DATATYPE_OBJECT_AS_STRING);

    // Initializes a new index
    m.def("init",
        [](py::object data, const std::string & method, const std::string & space,
                DistType dtype, DataType data_type, py::object space_params) {
            if (data_type != DATATYPE_DENSE_VECTOR) {
                throw std::runtime_error("Only Dense Vectors are currently supported");
            }

            py::object ret = py::none();
            switch (dtype) {
                case DISTTYPE_FLOAT: {
                    auto index = new IndexWrapper<float>(method, space, data, space_params);
                    ret = py::cast(index);
                    break;
                }
                case DISTTYPE_DOUBLE: {
                    auto index = new IndexWrapper<double>(method, space, data, space_params);
                    ret = py::cast(index);
                    break;
                }
                case DISTTYPE_INT: {
                    auto index = new IndexWrapper<int>(method, space, data, space_params);
                    ret = py::cast(index);
                    break;
                }
                default:
                    // should never happen
                    throw std::invalid_argument("Invalid DistType");
            }
            return ret;
        },
        py::arg("data"),
        py::arg("method") = "hnsw",
        py::arg("space") = "cosinesimil",
        py::arg("dtype") = DISTTYPE_FLOAT,
        py::arg("data_type") = DATATYPE_DENSE_VECTOR,
        py::arg("space_params") = py::none(),
        "This function initializes a new NMSLIB index\n\n"
        "Parameters\n"
        "----------\n"
        "data: array_like\n"
        "    A 2D array to create the index from. Each row in the matrix will be indexed as a\n"
        "    datapoint\n"
        "dtype: nmsbind.DistType optional\n"
        "    The type of index to create (float/double/int)\n"
        "data_type: nmsbind.DataType optional\n"
        "    The type of data to index (dense/sparse/string vectors)\n"
        "method: str optional\n"
        "    The index method to use\n"
        "space: str optional\n"
        "    The metric space to create for this index\n"
        "\n"
        "Returns\n"
        "----------\n"
        "    A new NMSLIB Index.\n");

    // Export Different Types of NMS Indices and spaces
    // hiding in a submodule to avoid cluttering up main namespace
    py::module dist_module = m.def_submodule("dist",
        "Contains Indexes and Spaces for different Distance Types");
    exportIndex<int>(&dist_module);
    exportIndex<float>(&dist_module);
    exportIndex<double>(&dist_module);
    return m.ptr();
}

template <typename dist_t>
void exportIndex(py::module * m) {
    // Export the index
    std::string index_name = distName<dist_t>() + "Index";
    py::class_<IndexWrapper<dist_t>>(*m, index_name.c_str())
        .def("createIndex", &IndexWrapper<dist_t>::createIndex,
             py::arg("index_params") = py::none(),
            "Creates the index, and makes it available for querying\n\n"
            "Parameters\n"
            "----------\n"
            "index_params: dict optional\n"
            "    Dictionary of optional parameters to use in indexing\n")

        .def("knnQuery", &IndexWrapper<dist_t>::knnQuery,
            py::arg("vector"), py::arg("k") = 10,
            "Finds the approximate K nearest neighbours of a vector in the index \n\n"
            "Parameters\n"
            "----------\n"
            "vector: array_like\n"
            "    A 1D vector to query for.\n"
            "k: int optional\n"
            "    The number of neighbours to return\n"
            "\n"
            "Returns\n"
            "----------\n"
            "ids: array_like.\n"
            "    A 1D vector of the ids of each nearest neighbour.\n"
            "distances: array_like.\n"
            "    A 1D vector of the distance to each nearest neigbhour.\n")

        .def("loadIndex", &IndexWrapper<dist_t>::loadIndex,
            py::arg("filename"),
            "Loads the index from disk\n\n"
            "Parameters\n"
            "----------\n"
            "filename: str\n"
            "    The filename to read from\n")

        .def("saveIndex", &IndexWrapper<dist_t>::saveIndex,
            py::arg("filename"),
            "Saves the index to disk\n\n"
            "Parameters\n"
            "----------\n"
            "filename: str\n"
            "    The filename to save to\n")

        .def("setQueryTimeParams",
            [](IndexWrapper<dist_t> * self, py::object params) {
                self->index->SetQueryTimeParams(loadParams(params));
            }, py::arg("params") = py::none(),
            "Sets parameters used in knnQuery.\n\n"
            "Parameters\n"
            "----------\n"
            "params: dict\n"
            "    A dictionary of params to use in querying. Setting params to None will reset\n")

        .def("__repr__", &IndexWrapper<dist_t>::repr);
}

DistType distTypeFromObject(py::object data) {
    if (py::isinstance<py::array_t<double>>(data)) {
        return DISTTYPE_DOUBLE;
    } else if (py::isinstance<py::array_t<float>>(data)) {
        return DISTTYPE_FLOAT;
    } else if (py::isinstance<py::array_t<int>>(data)) {
        return DISTTYPE_INT;
    }
    throw std::invalid_argument("unknown data type");
}

template <typename T>
ObjectVector loadObjectVector(py::object data) {
    if (!py::isinstance<py::array_t<T>>(data)) {
        throw std::invalid_argument("numpy dtype should match dist_type");
    }

    py::array_t<T> input(py::cast<py::array_t<T>>(data));

    auto buffer = input.request();
    if (buffer.ndim != 2) throw std::runtime_error("data must be a 2d array");

    ObjectVector ret;
    size_t rows = buffer.shape[0], features = buffer.shape[1];
    for (size_t row = 0; row < rows; ++row) {
        ret.push_back(new Object(row, -1, features * sizeof(T), input.data(row)));
    }
    return ret;
}

template <> std::string distName<int>() { return "Int"; }
template <> std::string distName<float>() { return "Float"; }
template <> std::string distName<double>() { return "Double"; }

AnyParams loadParams(py::object o) {
    AnyParams ret;
    if (o.is_none()) {
        return ret;
    }

    if (py::isinstance<py::dict>(o)) {
        py::dict items(o);
        for (auto & item : items) {
            std::string key = py::cast<std::string>(item.first);
            auto & value = item.second;

            // allow param values to be string/int/double
            if (py::isinstance<py::int_>(value)) {
                ret.AddChangeParam(key, py::cast<int>(value));
            } else if (py::isinstance<py::float_>(value)) {
                ret.AddChangeParam(key, py::cast<double>(value));
            } else if (py::isinstance<py::str>(value)) {
                ret.AddChangeParam(key, py::cast<std::string>(value));
            } else {
                std::stringstream err;
                err << "Unknown type for parameter '" << key << "'";
                throw std::invalid_argument(err.str());
            }
        }
    } else {
        throw std::invalid_argument("Can't convert to params");
    }
    return ret;
}
}  // namespace similarity
