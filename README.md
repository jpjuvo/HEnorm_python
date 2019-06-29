### Fork of [H&E histopathological staining normalization of schaugf](https://github.com/schaugf/HEnorm_python)
- Optimized for speed and multiprocessing - June '19 [Joni Juvonen](https://github.com/jpjuvo)  
- To recompile cython binaries, run ```python .\setup.py build_ext --inplace```

-----------------------------------

# Staining Unmixing and Normalization in Python

NOTE: Originally published at the Assessment of Mitosis Detection Algorithms 2013 challenge website (http://http://amida13.isi.uu.nl).

Adapted from original implementation (https://github.com/mitkovetta/staining-normalization).

One of the major difficulties in histopathology image analysis is apperance variability. 
For example, when performing mitosis detection, many false positives can arise when the histopathology slide is overstained. 
This Python code, adapted from the original MATLAB implementation, performs staining unmixing (separation of the hematoxylin and eosing stains) and apperance normalization. 
It is based on the method described in [1]. Some examples of staining normalization can be seen in the figure below.

[1] A method for normalizing histology slides for quantitative analysis, M Macenko, M Niethammer, JS Marron, D Borland, JT Woosley, G Xiaojun, C Schmitt, NE Thomas, IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250

# Examples

## Original Images

<img src='imgs/example1.png' width='30%'>
<img src='imgs/example2.png' width='30%'>

## Normalized Images

<img src='normalized/example1.png' width='30%'>
<img src='normalized/example2.png' width='30%'>

## Stain Unmixing

### Example 1

<img src='normalized/example1_H.png' width='30%'>
<img src='normalized/example1_E.png' width='30%'>

### Example 2

<img src='normalized/example2_H.png' width='30%'>
<img src='normalized/example2_E.png' width='30%'>



