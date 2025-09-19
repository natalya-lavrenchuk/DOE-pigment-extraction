# Design of Experiments (DOE) – Box-Behnken Design for Pigment Extraction  

This repository contains my work on a **Design of Experiments (DOE)** project, developed for the course *Machine Learning for Process Engineering (SP25)* at the University of Padova.  

The objective was to apply a **Box-Behnken design** (three factors, three levels) to study and optimize the extraction of natural pigments (betacyanin and betaxanthin) from prickly pear. The DOE was analyzed using both **MATLAB** and **Minitab**, with response surface methodology (RSM) and ANOVA used to interpret the results.  

---

## Repository Contents  

- **DOE_assignment_description.pdf**  
  Original assignment description provided by the professor, outlining the DOE problem and setup.  

- **DOE_dataset.mat**  
  MATLAB dataset containing experimental data for the Box-Behnken design, including input factors and measured pigment concentrations.  

- **lavrenchuk_MLfPE_DOE.m**  
  MATLAB script used to analyze the DOE data. Implements regression models, builds response surfaces, and generates plots of factor interactions.  

- **lavrenchuk_MLfPE_DOE.mat**  
  MATLAB output file containing numerical solutions, fitted model coefficients, and prediction results.  

- **lavrenchuk_MLfPE_DOE.mpx**  
  Minitab project file used for DOE analysis. Includes the Box-Behnken design matrix, ANOVA, and surface plots.  

- **report_MLfPE_DOE.pdf**  
  Final report discussing the DOE methodology, experimental design, statistical analysis (ANOVA), and optimal factor settings for pigment extraction.  


---

## Methods  

- **Design type:** Box-Behnken (3 factors × 3 levels).  
- **Analysis tools:** MATLAB + Minitab.  
- **Statistical methods:** Regression, ANOVA, Response Surface Methodology (RSM).  
- **Goal:** Identify factor combinations that maximize pigment yield while analyzing significance of main effects and interactions.  

---

## Acknowledgements  

- Course: *SP25 Machine Learning for Process Engineering*  
- Instructor: Prof. Pierantonio Facco  
- University of Padova  
