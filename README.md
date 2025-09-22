
# Building a new emulator 搭建一个新的Emulator

1. Duplicate the folder of '2025_Bristol_5D_v002_Xin/modlowice_temp' and rename the folder with **the name of emulator**
2. Clean the file in 'emulator/' and 'training_data/'
3. **Edit step1.** to read in raw training data and formating them. Then run.
4. *if needed. Edit hyperparameters in step2. Then run.
5. *optional. ca delete the data in training_data/

Finilizing: success if there're 3 files be created in emulator/: emul_in_X_5variables_mean_std.nc emul_in_Y_PCA.nc GPList.h5\

Now the emulator is ready to use!

--------------------------

# Use an emulator

1. Prepare the data: A example [[test_data.1.res]]. The data must be [5xN] contains a head like: \
co2 obliquity esinw ecosw ice \
442.70 23.44 0.016279 -0.003734 0.00\
... ...
2. Enter the folder 'prediction/'
3. Edit run.ipynb. \
    3.1 indicate the name of emulator you wanna use (name of emulator is the folder name of emulators)\
    3.2 indicate the forcing data you wanna use to predict
4. Then run.

If there's netcdf file and gif file be created in the prediction folder. Then success!