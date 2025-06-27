# Human Intracranial Single Unit Sorting
[![DOI](https://zenodo.org/badge/891756132.svg)](https://doi.org/10.5281/zenodo.14217838)

Cognitive neuroscience rapidly evolved following the possibility to record single neurons in humans.

To date, the recordings comes from medical procedures in which implantation of the electrode into the human brain is a necessary part of the clinical treatment, For example, invasive epilepsy monitoring using the so-called “Behnke-Fried” electrodes.

The data recorded from micro-electrodes have the potential to review how human brain functions with unprecedented high spatial and time resolution. The opportunity brought by micro-electrode recordings hinges heavily on the sorting algorithms. The main stream of sorting algorithms either use threshold to do channel-wise sorting (for example, Osort) or make use of relative spatial distribution to adjust for drift (Kilosort). In the event where we have no information of spatial information of the channels, for example, electrical signals recorded in “Behnke-Fried” electrodes, it is more efficient to do single-channel sorting.

The extant single channel-level sorting can benefit from the advanced in clustering algorithms and techniques for ensuring robustness against artifacts, which are often present in human single unit recordings. This novel algorithm aims to build upon the current single channel-level sorting algorithm, and apply template-matching to more robustly sort single unit, in comparison the thresholding.

The main file is written in Matlab, called, 'template matching and learning.m'. An example script of how to use the function on BCI2000 data has been provided. 
