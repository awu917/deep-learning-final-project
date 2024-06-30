# Deep Learning Supernova Classifiers 
AJ, Arin, Katie, Stella 

To run our RNN model, run train.py with either "vanilla," "lstm," or "gru" passed into model_type. 
To run our transformer model, run transformer.py 

Results: Our best validation accuracy of 0.9770 was achieved with the LSTM model after 50 epochs. 

Future large, wide-field photometric surveys of our universe will produce vast amounts of data pertaining to supernovae, having the potential to revolutionize the field of cosmology. Using machine learning methods will become necessary to identify and classify supernovae from such surveys, as this boom in data will make it impossible for astronomers to classify supernovae by hand. The recently completed Dark Energy Survey itself has even used machine learning techniques to classify type-Ia supernovae.

We are especially interested in the binary classification task of classifying type-Ia supernovae vs. non-type-Ia. Type Ia supernovae are especially exciting to the field of dark energy cosmology. This is because type-Ia supernovae are unique in that they are standard candles, meaning they have a known luminosity. That luminosity can be used to measure the supernova’s distance to the Earth. By comparing supernovae distances to their redshifts, astronomers Saul Perlmutter, Brian Schmidt, and Adam Riess discovered that high redshift supernovae were fainter than expected— indicating that the expansion of the universe is accelerating.

Because timeseries photometric data is required to classify supernovae, we use recurrent networks and transformers to classify supernovae as type-Ia vs non-type-Ia based on their light curves.
Data is sourced from the Supernovae Photometric Classification Challenge, a set of 21k simulated light curves. The dataset consists of 21,319 simulated supernova light curves. Each supernovae sample consists of a timeseries of flux measurements, with errors, in the g, r, i, z bands (one band for each time step), along with the position on the sky and dust extinction. All models were trained on 50% of the dataset, with the other 50% used for validation. 

![Poster](https://drive.google.com/uc?export=view&id=131fjrnjislnbPknBS0DVKsZXKLMvXwxR)
