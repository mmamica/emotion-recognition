# EEG-Based Emotion Recognition with Convolutional Neural Networks
The aim of the project is to detect emotions - valence and arousal - based on EEG signals. Machine learning techinques were used to obtain high accuracy. The project is a replication of the method decribed in the <a href="https://link.springer.com/chapter/10.1007/978-981-15-1899-7_11" target="_blank">**"Convolutional Neural Networks on EEG-Based Emotion Recognition"**</a> article propsed by Chunbin Li et al.

### Dataset
For the project <a href="https://link.springer.com/chapter/10.1007/978-981-15-1899-7_11" target="_blank">**DEAP** </a> dataset was used. It contains EEG data acquired from 32 participants. It can be downloaded by sending a request to authorised personel.

### Preprocessing

* Extraction of averaged power spectral densities for each channel and each frequency band with the use of <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html">**periodogram** </a>.
* Azimuthal Equidistant Projection was performed on geographical coordinates of sensors to retrieve appropriate geometrical coordinates in 2D space. 
* Clough-Tocher scheme was applied to interpolate averaged PSDs of electrodes and estimate values in between.
