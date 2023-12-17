# wyGPT
No model is perfect but some are useful.

***Introduction***

This is my 2.5 years' day-and-night efforts on GPT. It is mature and highly optimized on single GPU.

***usage:***

make

./train text_file.txt

./gpu "prompt"

./cpu -t 2 "prompt"

***a working version trained on PubMed***

Link: https://pan.baidu.com/s/1d3PQahvWi5SARUDzsm_-Uw?pwd=qn5f 
Password: qn5f 

To finetune it (12 hours per iteration), type:

```./train your_text.txt -i model -s 44032```

***sample text***

```The EGFR gene mutation status of NSCLC patients was analyzed by Sanger sequencing. A total of 17 patients (52.2%) were positive for at least one EGFR mutation. Among them, 11 patients (72.7%) had an EGFR gene mutation. The majority of patients were treated with EGFR-TKIs, of which one patient did not receive any EGFR TKIs. In addition, the median progression-free survival (PFS) was 9.0 months (95% CI, 7.2-10.4 months) and the median overall survival (OS) was 21.5 months (95% CI, 19.4-23.3 months). In a multivariate Cox proportional hazards model, only the EGFR mutation status was an independent prognostic factor for PFS (hazard ratio, 0.31; 95% CI, 0.18-0.53; P = 0.0002). The EGFR mutation sequencing from NSCLC patients is a cost-effective screening procedure for detecting EGFR mutations in the early stages of NSCLC. Our findings may provide a solution for patients with NSCLC who may be considered for aggressive treatment. Should the EGFR gene mutations be considered as an independent risk factor for PFS and OS in patients with advanced NSCLC, this may be an important target for appropriate management. (JINSON Pharmaceuticals, Inc., San Diego, CA). Department of Oncology, The University of Texas MD Anderson Cancer Center, Houston, TX. (Tracking ID #216733). BACKGROUND: Carcinoma of the prostate is a rare malignancy and poorly differentiated tumor of the ```
