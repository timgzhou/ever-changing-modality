# ever-changing-modality
Ever-Adapting Network for Ever-Changing Modality setting

To download the Dataset used for the experiments:

1. Download GeoBench dataset by:
```
pip install geobench --no-deps
pip install h5py
export GEO_BENCH_DIR='/home/...' (where you want the datasets to be)
```
2. Download EuroSAT directly from TorchGeo

3. download reBEN by 
```
cd datasets
wget https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1
wget https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1
```
