from setuptools import setup

setup(
    name         = "NeuralCompresion_v3",
    version      = "0.0.1.dev",
    author       = "Yi Huang",
    author_email = "yhuang2@bnl.gov",
    description  = ("Time projection chamber data compression "
                    "with bicephalous convolution autoencoder "
                    "enabling variable compression ratio for "
                    "Sparse input"),
    license      = "BSD 3-Clause 'New' or 'Revised' License",
    keyword      = ("autoencoder, data compression, sparse data, "
                    "variable compression ratio"),
    packages     = ['neuralcompress_v3'],
    classifiers  = ["Development Status :: 3 - Alpha",
                    ("Topic :: Scientific/Engineering :: "
                     "Artificial Intelligence"),
                    ("License :: OSI Approved :: BSD 3-Clause 'New' "
                     "or 'Revised' License")],
)
