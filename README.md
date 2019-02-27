# MTGDB models
Scripts and tools for training models based on MTGDB datasets.

## Gaia classifiers
A set of SVM classifiers trained on the in-house MTG collection using Gaia and Essentia rutines.  

### Use

1. Compile and install [Gaia](https://github.com/MTG/gaia) with python wrappers. Use the code from the following [pull request](https://github.com/MTG/gaia/pull/86) as it is the version compatible with the updated Essentia features.  
2. Compile and install [Essentia](https://github.com/MTG/essentia). This repository relies on the `batch_music_extractor` functionality which is temporarily misplaced in an outdated package in the master branch of Essentia. The correctly located module can be found on this [pull request](https://github.com/MTG/essentia/pull/833).
3. In `src/gaia_classifiers/path_config.py` set the variable `MTGDB_DIR` to match your MTGDB mount point. Here you can also bypass the models that you do not want to train by commenting out the respective lines.
4. Optionally modify `src/gaia_classifiers/run.sh` to choose which Python interpreter to use and where to store the generated data. However right now Gaia is not available for Python 3.
5. Run `./src/gaia_classifiers/run.sh` 


*Note: This documentation must be updated after merging the aforementioned pull requests to master.*