| **CF Method**       | **Dataset [Models]**      | _**Validity**_ | _**Sparsity**_   | _**Proximity**_  | _**Generation Time (secs.)**_ |
|---------------------|---------------------------|----------------|------------------|------------------|-------------------------------|
| Nearest-CT          | _Breast Cancer_ [RF, MLP] | [$1,1$]        | [$29.98, 29.52$] | [$18.05, 10.78$] | [$0.10, 0.08$]                |
| Nearest-CT          | _Diabetes_ [AdaBoost]     | [$0$]          | [inf]            | [inf]            | [inf]                         |
| Nearest-CT          | _Sonar_ [MLP]             | [$1$]          | [$59.76$]        | [$30.33$]        | [$0.12$]                      |
| Nearest-CT          | _Wave_ [XGBoost, MLP]     | [$1,1$]        | [$20.86, 20.83$] | [$14.03, 12.02$] | [$0.08,0.08$]                 |
| Nearest-CT          | _Boston Housing_ [MLP-Reg]| [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| FeatTweak           | _Breast Cancer_ [RF, MLP] | [$0.70$, N/A]  | [$2.42$, N/A]    | [$\mathbf{4.20}$, N/A]| [$2200$, N/A]            |
| FeatTweak           | _Diabetes_ [AdaBoost]     | [N/A]          | [N/A]            | [N/A]            | [N/A]                         |                 
| FeatTweak           | _Sonar_ [MLP]             | [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| FeatTweak           | _Wave_ [XGBoost, MLP]     | [N/A, N/A]     | [N/A, N/A]       | [N/A, N/A]       | [N/A, N/A]                    |
| FeatTweak           | _Boston Housing_ [MLP-Reg]| [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| FOCUS               | _Breast Cancer_ [RF, MLP] | [$0.66$, N/A]  | [$2.60$, N/A]    | [$4.39$, N/A]    | [$\mathbf{120}$, N/A]         |
| FOCUS               | _Diabetes_ [AdaBoost]     | [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| FOCUS               | _Sonar_ [MLP]             | [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| FOCUS               | _Wave_ [XGBoost, MLP]     | [N/A, N/A]     | [N/A, N/A]       | [N/A, N/A]       | [N/A, N/A]                    |
| FOCUS               | _Boston Housing_ [MLP-Reg]| [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| DeepFool            | _Breast Cancer_ [RF, MLP] | [N/A, $0.85$]  | [N/A, $29.70$]   | [N/A, $10.30$]   | [N/A, $0.02$]                 |
| DeepFool            | _Diabetes_ [AdaBoost]     | [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| DeepFool            | _Sonar_ [MLP]             | [$0.79$]       | [$60$]           | [$9.97$]         | [$0.03$]                      |
| DeepFool            | _Wave_ [XGBoost, MLP]     | [N/A, $0.88$]  | [N/A, $20.30$]   | [N/A, $16.60$]   | [N/A, $0.02$]                 |
| DeepFool            | _Boston Housing_ [MLP-Reg]| [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| GRACE               | _Breast Cancer_ [RF, MLP] | [N/A, $0.69$]  | [N/A, $2.07$]    | [N/A, $5.94$]    | [N/A, $\mathbf{1.20}$]        |
| GRACE               | _Diabetes_ [AdaBoost]     | [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| GRACE               | _Sonar_ [MLP]             | [$0.73$]       | [$3.91$]         | [$8.27$]         | [$\mathbf{2.80}$]             |
| GRACE               | _Wave_ [XGBoost, MLP]     | [N/A, $0.54$]  | [N/A, $3.09$]    | [N/A, $7.11$]    | [N/A, $\mathbf{1.50}$]        |
| GRACE               | _Boston Housing_ [MLP-Reg]| [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| DiCE                | _Breast Cancer_ [RF, MLP] | [X,Y]          |  [X,Y]           | [X,Y]            | [X,Y]                         |
| DiCE                | _Diabetes_ [AdaBoost]     | [X,Y]          |  [X,Y]           | [X,Y]            | [X,Y]                         |
| DiCE                | _Sonar_ [MLP]             | [X,Y]          |  [X,Y]           | [X,Y]            | [X,Y]                         |
| DiCE                | _Wave_ [XGBoost, MLP]     | [X,Y]          |  [X,Y]           | [X,Y]            | [X,Y]                         |
| DiCE                | _Boston Housing_ [MLP-Reg]| [X,Y]          |  [X,Y]           | [X,Y]            | [X,Y]                         |
| LORE                | _Breast Cancer_ [RF, MLP] | [$0.65,0.58$]  | [$\mathbf{2.05},\mathbf{2.02}$] | [$4.63,5.63$] | [$2200,2100$]     |
| LORE                | _Diabetes_ [AdaBoost]     | [$0.52$]       | [$1.61$]         | [$4.76$]         | [$1900$]                      |
| LORE                | _Sonar_ [MLP]             | [$0.56$]       | [$3.35$]         | [$7.36$]         | [$2700$]                      |
| LORE                | _Wave_ [XGBoost, MLP]     | [$0.68,0.56$]  | [$2.74,3.19$]    | [$6.60,6.41$]    | [$2000,1800$]                 |
| LORE                | _Boston Housing_ [MLP-Reg]| [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| MACE                |  _Breast Cancer_ [RF, MLP]| [$0.75$, N/A]  | [$2.58$, N/A]    | [$4.47,$ N/A]    | [$2280$, N/A]                 |
| MACE                | _Diabetes_ [AdaBoost]     | [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| MACE                | _Sonar_ [MLP]             | [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| MACE                | _Wave_ [XGBoost, MLP]     | [N/A, N/A]     | [N/A, N/A]       | [N/A, N/A]       | [N/A, N/A]                    |
| MACE                | _Boston Housing_ [MLP-Reg]| [N/A]          | [N/A]            | [N/A]            | [N/A]                         |
| ReLAX-Global        |  _Breast Cancer_ [RF, MLP]| [$0.70,0.75$]  | [$2.39,2.14$]    | [$4.46,5.92$]    | [$2100,1200$]                 |
| ReLAX-Global        | _Diabetes_ [AdaBoost]     | [$0.70$]       | [$1.50$]         | [$\mathbf{4.41}$]| [$2000$]                      |
| ReLAX-Global        | _Sonar_ [MLP]             | [$0.80$]       | [$\mathbf{2.79}$]| [$\mathbf{7.32}$]| [$1400$]                      |
| ReLAX-Global        |  _Wave_ [XGBoost, MLP]    | [$0.84,0.83$]  | [$\mathbf{2.62},\mathbf{2.69}$] | [$\mathbf{5.93},\mathbf{6.38}$] | [$1300,1200$] |
| ReLAX-Global        | _Boston Housing_ [MLP-Reg]| [$0.74$]       | [$\mathbf{2.41}$]| [$\mathbf{5.10}$]| [$1300$]                      |
| ReLAX-Local         |  _Breast Cancer_ [RF, MLP]| [$\mathbf{0.78},\mathbf{0.84}$]   | [$2.57,2.22$]    | [$4.49,\mathbf{5.87}$] | [$1900,1100$]      |
| ReLAX-Local         | _Diabetes_ [AdaBoost]     | [$\mathbf{0.76}$] | [$\mathbf{1.49}$] | [$3.60$]     | [$\mathbf{1800}$]             |
| ReLAX-Local         | _Sonar_ [MLP]             | [$\mathbf{0.97}$] | [$3.04$]          | [$7.66$]     | [$1000$]                      |
| ReLAX-Local         |  _Wave_ [XGBoost, MLP]    | [$\mathbf{0.88},\mathbf{0.91}$] | [$2.67,2.71$]  | [$6.02,6.50$] | [$\mathbf{1100},1340$]          |
| ReLAX-Local         | _Boston Housing_ [MLP-Reg]| [$\mathbf{0.81}$] | [$2.57$]    | [$5.36$]       | [$\mathbf{1000}$]                 |
