## [ICASSP 2026] WINDMOE: MIXTURE-OF-EXPERTS METHOD FOR WIND POWER FORECASTING UNDER EXTREME WEATHER CONDITIONS

> **Authors:**

Lei Liu, Qi Wang, Hongwei Zhao, Ruibo Guo, Jiahui Huang, Tengyuan Liu, Bin Li.

Our paper has been accepted by ICASSP 2026.

Website：

## 1. Abstract

Wind power forecasting (WPF) plays a crucial role in grid management. However, existing WPF methods that rely on a single forecasting model, often struggle to adapt to extreme weather conditions while maintaining overall performance. To address this challenge, this paper introduces WindMoE, a Mixture-of-Experts (MoE) based model for WPF. Specifically, WindMoE comprises three core components: a general expert learns from the entire dataset to ensure performance on normal conditions, a specialized expert is trained using a novel loss re-weighting strategy to focus on extreme conditions, and a learnable router fuses their predictions based on adaptive gating mechanism. The MoE framework offers the advantage of assigning different experts to focus on distinct weather scenarios, enabling more specialized and accurate forecasting. Experimental results show that WindMoE achieves state-of-the-art performance, with a notable RMSE reduction of up to 5.4% in typhoons and 4.4% in cold waves.

## 2. Additional Experientment Results

We conduct additional experiments to further verify the effectiveness of WindMoE. First, we add several baselines, SegRNN[1] and MICN[2]. The results were shown in following tables.

$$
\begin{array}{l|cc|cc|cc|cc}
\hline 
\text{Datasets} & \text{Goldwind} & \text{Goldwind} & \text{Goldwind} & \text{Goldwind} & \text{Jilin} & \text{Jilin} & \text{Jilin} & \text{Jilin} \\
\hline 
\text{station} & 225 & 225 & 420 & 420 & 57 & 57 & 58 & 58 \\
\hline 
\text{Metrics} & \text{MAE} & \text{RMSE} & \text{MAE} & \text{RMSE} & \text{MAE} & \text{RMSE} & \text{MAE} & \text{RMSE} \\
\hline 
\text{DLinear} & 0.11382 & 0.16930 & 0.08513 & 0.13094 & 0.15877 & 0.19306 & 0.23369 & 0.27902 \\
\text{PatchTST} & 0.23532 & 0.27967 & 0.17020 & 0.21500 & 0.18996 & 0.22566 & 0.27626 & 0.32391 \\
\text{GAN} & 0.10474 & 0.16004 & 0.07993 & 0.12568 & 0.11398 & 0.16213 & 0.14017 & 0.19697 \\
\text{Transfer} & 0.30252 & 0.37461 & 0.20560 & 0.23807 & 0.18906 & 0.23446 & 0.29471 & 0.34880 \\
\text{SegRNN} & 0.11458 & 0.16782 & 0.08794 & 0.13284 & 0.13439 & 0.17669 & 0.18455 & 0.23541 \\
\text{MICN} & 0.12616 & 0.17739 & 0.08916 & 0.13238 & 0.14568 & 0.18249 & 0.20595 & 0.25236 \\
\mathbf{WindMoE} & \mathbf{0.10432} & \mathbf{0.15294} & \mathbf{0.07968} & \mathbf{0.12232} & \mathbf{0.11228} & \mathbf{0.15811} & \mathbf{0.13720} & \mathbf{0.18976} \\
\hline
\end{array}
$$

$$
\begin{array}{l|cc|cc|cc|cc}
\hline 
\text{Datasets} & \text{Goldwind} & \text{Goldwind} & \text{Goldwind} & \text{Goldwind} & \text{Fujian} & \text{Fujian} & \text{Fujian} & \text{Fujian} \\
\hline 
\text{Station} & 402 & 402 & 1700 & 1700 & 15 & 15 & 18 & 18 \\
\hline 
\text{Metrics} & \text{MAE} & \text{RMSE} & \text{MAE} & \text{RMSE} & \text{MAE} & \text{RMSE} & \text{MAE} & \text{RMSE} \\
\hline 
\text{DLinear} & 0.02292 & 0.03363 & 0.07607 & 0.11057 & 0.17144 & 0.21988 & 0.14729 & 0.19705 \\
\text{PatchTST} & 0.06118 & 0.08129 & 0.28663 & 0.32326 & 0.32328 & 0.36850 & 0.25460 & 0.30198 \\
\text{GAN} & \mathbf{0.02288} & 0.03415 & \mathbf{0.07395} & 0.10870 & 0.08309 & 0.12331 & 0.07487 & 0.11218 \\
\text{Transfer} & 0.06537 & 0.07670 & 0.30388 & 0.34803 & 0.31237 & 0.36553 & 0.22281 & 0.27452 \\
\text{SegRNN} & 0.02312 & 0.03389 & 0.07696 & 0.11230 & 0.12547 & 0.16823 & 0.13394 & 0.17611 \\
\text{MICN} & 0.02442 & 0.03612 & 0.08620 & 0.12577 & 0.14751 & 0.19825 & 0.11832 & 0.16558 \\
\mathbf{WindMoE} & 0.02292 & \mathbf{0.03359} & \mathbf{0.07395} & \mathbf{0.10740} & \mathbf{0.08070} & \mathbf{0.11754} & \mathbf{0.07221} & \mathbf{0.10609} \\
\hline
\end{array}
$$

Additionally, we test our MoE structure on DLinear to demonstrate the compatibility with different forecasting models. We adopt DLinear instead of iTransformer and the results were shown in following tables.

$$
\begin{array}{l|lcc}
\hline
\textbf{Weather} & \textbf{Method} & \textbf{MAE} & \textbf{RMSE} \\ 
\hline
\text{Cold wave} & \text{GenExp-only} & 0.15877 & 0.19306 \\
& \text{SpeExp-only} & 0.19537 & 0.22507 \\
& \text{No-Router} & 0.19453 & 0.22357 \\
& \mathbf{WindMoE} & \mathbf{0.14870} & \mathbf{0.17928} \\ 
\hline
\text{Typhoon} & \text{GenExp-only} & 0.17144 & 0.21988 \\
& \text{SpeExp-only} & 0.28044 & 0.32770 \\
& \text{No-Router} & 0.20700 & 0.23811 \\
& \mathbf{WindMoE} & \mathbf{0.13577} & \mathbf{0.17430} \\ 
\hline
\end{array}
$$

[1] Lin S, Lin W, Wu W, et al. Segrnn: Segment recurrent neural network for long-term time series forecasting[J]. arXiv preprint arXiv:2308.11200, 2023.

[2] Wang H, Peng J, Huang F, et al. Micn: Multi-scale local and global context modeling for long-term series forecasting[C]//The eleventh international conference on learning representations. 2023.

## 3. Datasets

The data preprocessing code is provided.

## 4. Usage

- an example for train and evaluate a new model：
  You can set the parameters in a bash file.

```bash
python run.py
```



## 5. Citation

If you find our work useful in your research, please consider citing:

```latex

```

If you have any problems, contact me via liulei13@ustc.edu.cn.
