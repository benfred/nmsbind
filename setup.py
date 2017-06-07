import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

__version__ = '0.0.5'

libdir = os.path.join(".", "nmslib", "similarity_search")
library_file = os.path.join(libdir, "release", "libNonMetricSpaceLib.a")
source_files = ['nmsbind.cc']
extra_objects = []

if os.path.exists(library_file):
    # if we have a prebuilt nmslib library file, use that.
    extra_objects.append(library_file)

else:
    # Otherwise build all the files here directly (excluding extras which need eigen/boost)
    exclude_files = set("""lsh.cc lsh_multiprobe.cc lsh_space.cc falconn.cc nndes.cc space_sqfd.cc
                        dummy_app.cc main.cc""".split())

    for root, subdirs, files in os.walk(os.path.join("nmslib", "similarity_search", "src")):
        source_files.extend(os.path.join(root, f) for f in files
                            if f.endswith(".cc") and f not in exclude_files)

ext_modules = [
    Extension(
        'nmsbind',
        source_files,
        include_dirs=['%s/include' % libdir],
        language='c++',
        extra_objects=extra_objects,
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/openmp', '/O2'],
        'unix': ['-O3',],
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        c_opts['unix'].append("-fopenmp")
        link_opts['unix'].append('-fopenmp')

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        # extend include dirs here (don't assume numpy/pybind11 are installed when first run, since
        # pip could have installed them as part of executing this script
        import pybind11
        import numpy as np
        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))
            ext.include_dirs.extend([
                # Path to pybind11 headers
                pybind11.get_include(),
                pybind11.get_include(True),

                # Path to numpy headers
                np.get_include()
            ])

        build_ext.build_extensions(self)


setup(
    name='nmsbind',
    version=__version__,
    author='Ben Frederickson',
    author_email='ben@benfrederickson.com',
    url='https://github.com/benfred/nmsbind',
    description='Alternate Python Bindings for NMSLIB',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.0', 'numpy'],
    cmdclass={'build_ext': BuildExt},
    test_suite="tests",
    zip_safe=False,
)
