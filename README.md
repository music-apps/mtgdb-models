# MTGDB models
Scripts and tools for training models based on MTGDB datasets.

## Gaia classifiers
A set of SVM classifiers trained on the in-house MTG collection using Gaia and Essentia rutines.  

### Use

1. Compile and install [Gaia](https://github.com/MTG/gaia) with python wrappers. Use the code from the following [pull request](https://github.com/MTG/gaia/pull/86) as it is the version compatible with the updated Essentia features.  
2. Get [Essentia](https://github.com/MTG/essentia) and make sure it is accesible from the same python interpreter as Gaia (i.e, you should be able to import both modules on an iPython session).
3. In `src/gaia_classifiers/path_config.py` set the variable `MTGDB_DIR` to match your MTGDB mount point. Here you can also bypass the models that you do not want to train by commenting out the respective lines.
4. Optionally modify `src/gaia_classifiers/run.sh` to choose which Python interpreter to use and where to store the generated data. However right now Gaia is not available for Python 3.
5. Run `./src/gaia_classifiers/run.sh` 


*Note: This documentation must be updated after merging the aforementioned pull requests to master.*
