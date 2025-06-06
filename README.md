
# Customer Personality Clustering

## About This Project

This project focuses on the in-depth analysis and clustering of a company's customers based on their behavioral, demographic, and purchase history data. The main goal is to identify consumption patterns and distinct characteristics of each customer group, enabling the development of more targeted and personalized marketing strategies.

## Key Features and Project Stages

* **Data Preprocessing & Feature Engineering:**
    * **Temporal Data Management:** Converting the customer registration date (`Dt_Customer`) to a datetime data type and calculating new temporal features such as "Customer Duration" (`Customer_For`).
    * **Age Calculation:** Precisely calculating customer ages based on their birth year (`Year_Birth`) and the latest registration date in the dataset.
    * **Total Spending Calculation (`Spent`):** Aggregating all customer expenditures across various product categories (e.g., wines, meat, gold products) to create a single `Spent` feature.
    * **Categorical Data Management:** Standardizing and mapping similar values in categorical features like `Education` and `Marital_Status` for simplification and improved data quality.
    * **Outlier Removal:** Identifying and removing abnormal data points (outliers) using the Interquartile Range (IQR) method to enhance analysis quality.
    * **Removal of Irrelevant/Redundant Features:** Dropping columns that are either redundant (their information is present in newly created features) or not useful for clustering analysis (such as customer ID, or columns with a single constant value).
    * **Standardization and Encoding:** Applying scaling (StandardScaler) to numerical features and ordinal encoding (OrdinalEncoder) to remaining categorical features to prepare the data for modeling.

* **Dimensionality Reduction:**
    * Utilizing **Principal Component Analysis (PCA)** to reduce the dataset's dimensions to 3 principal components (`PC1`, `PC2`, `PC3`). This facilitates easier visualization of clusters and improves the efficiency of clustering algorithms.

* **Clustering:**
    * **Implementation and Evaluation:**
        * **KMeans:** Implementing and evaluating the performance of this algorithm.
        * **Agglomerative Clustering:** Implementing and evaluating the performance of this algorithm.
        * **Gaussian Mixture Models (GMM):** Implementing and evaluating the performance of this algorithm.
    * **Determining Optimal Number of Clusters:** Employing various metrics to find the best number of clusters:
        * **Elbow Method with WCSS:** For KMeans.
        * **Silhouette Score:** To evaluate the quality of cluster separation and cohesion.
        * **Davies-Bouldin Index:** A metric for evaluating cluster compactness and separation (lower values are better).
        * **AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion):** For GMM, indicating a balance between model complexity and fit (lower values are better).

* **Cluster Analysis & Visualization:**
    * **Cluster Profiles:** Calculating the mean and median of key features such as income, spending, and number of web/store purchases for each cluster.
    * **Comparative Plots:** Visualizing the average income and spending for each cluster using bar plots.
    * **Interactive 3D Plots:** Displaying the identified clusters in the 3D PCA space using Plotly Express, allowing for interactive exploration of clusters and their centroids.
    * **Cluster Interpretation:** Providing interpretations of the behavioral and financial patterns of each cluster (e.g., "High-Income Savers," "Balanced Shoppers," "Big Spenders," and "Budget Shoppers") based on clustering results.

## Dataset

This project utilizes the `dataset.csv` file, which contains customer information including demographic details, income, registration date, and spending amounts across various product categories.

## Project Structure

```
customer_personality_clustering/
├── visualizations/                       # Folder for storing generated plots and visualizations
│   ├── bar_plot.png                      # Example of a comparative bar plot for clusters
│   ├── ...                               # Other plot images such as histograms, correlation heatmap, and clustering evaluation plots
├── customer_segmentation_clustering_en.ipynb  # Main Jupyter Notebook with code and analysis in English
├── customer_segmentation_clustering_fa.ipynb  # Main Jupyter Notebook with code and analysis in Persian
├── dataset.csv                           # The primary dataset used in the project
└── requirements.txt                      # A list of all Python libraries required for the project
```

## Technologies and Libraries Used

* **Python 3.x**
* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical operations.
* `scikit-learn`: A comprehensive set of machine learning tools for preprocessing, dimensionality reduction (PCA), and clustering algorithms (KMeans, AgglomerativeClustering, GaussianMixture).
* `matplotlib`: For creating static plots.
* `seaborn`: For advanced and aesthetically pleasing statistical visualizations.
* `plotly.express`: For creating interactive plots, especially 3D cluster visualizations.
* `warnings`: For managing warning messages.

## How to Set Up and Run

Follow these steps to set up and run this project:

1.  **Clone the repository:**
    First, clone the project repository from GitHub to your system:
    ```bash
    git clone [https://github.com/sorna-fast/customer_personality_clustering.git](https://github.com/sorna-fast/customer_personality_clustering.git)
    cd customer_personality_clustering
    ```

2.  **Create and activate a virtual environment:**
    It is highly recommended to create a virtual environment to manage dependencies:

    * **For Windows users:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **For macOS/Linux users:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    After activating the virtual environment, install all required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the notebooks:**
    To view and execute the analysis and code, start Jupyter Notebook or Jupyter Lab:

    * **Jupyter Lab:**
        ```bash
        jupyter lab
        ```
    * **Jupyter Notebook:**
        ```bash
        jupyter notebook
        ```
    After running one of the above commands, a page will open in your browser. You can open `customer_segmentation_clustering_en.ipynb` (English version) or `customer_segmentation_clustering_fa.ipynb` (Persian version) and run the cells sequentially to observe the analysis results and visualizations.

## Key Results and Insights

This project successfully identified several distinct customer clusters. Each cluster exhibits unique behavioral and financial patterns that can serve as a basis for targeted marketing strategies. Some potential customer segmentations (assuming optimal cluster count and interpretation of results):

* **High-Income Savers:** Customers with high income who tend to spend less on various products.
* **Balanced Shoppers:** Customers with moderate income and balanced spending across different product categories.
* **Big Spenders:** Customers who have high expenditures across various product categories, especially expensive ones.
* **Budget Shoppers:** Customers with lower income who spend less and focus their expenditures on specific categories.

These insights help businesses better understand their customer groups, create personalized marketing campaigns, and enhance customer experience.



👋 We hope you find this project useful! 🚀

## Contact Developer  
    Email: masudpythongit@gmail.com 
    Telegram: https://t.me/Fast_programmer
🔗 GitHub Profile: [sorna-fast](https://github.com/sorna-fast)

## License
This project is licensed under the [MIT](LICENSE) License.

---

# خوشه‌بندی شخصیت مشتریان (Customer Personality Clustering)

## درباره این پروژه

این پروژه به تحلیل عمیق و خوشه‌بندی مشتریان یک شرکت بر اساس داده‌های رفتاری، دموگرافیک و تاریخچه خرید آن‌ها می‌پردازد. هدف اصلی این است که با شناسایی گروه‌های متمایز مشتریان (خوشه‌ها)، الگوهای مصرف و ویژگی‌های منحصر به فرد هر گروه آشکار شود تا بتوان استراتژی‌های بازاریابی هدفمندتر و شخصی‌سازی شده‌تری را تدوین کرد.

## ویژگی‌ها و مراحل اصلی پروژه

* **پیش‌پردازش و مهندسی ویژگی (Data Preprocessing & Feature Engineering):**
    * **مدیریت داده‌های زمانی:** تبدیل فرمت تاریخ عضویت مشتری (`Dt_Customer`) به نوع داده زمانی و محاسبه ویژگی‌های جدید مانند "مدت زمان عضویت مشتری" (`Customer_For`).
    * **محاسبه سن:** محاسبه دقیق سن مشتریان بر اساس سال تولد (`Year_Birth`) و جدیدترین تاریخ ثبت در مجموعه داده.
    * **محاسبه کل هزینه‌ها (`Spent`):** جمع‌آوری تمام هزینه‌های مشتریان در دسته‌های مختلف محصولات (مانند نوشیدنی، گوشت، طلا و...) برای ایجاد یک ویژگی واحد.
    * **مدیریت داده‌های دسته‌بندی‌شده:** یکپارچه‌سازی و نگاشت (Mapping) مقادیر مشابه در ویژگی‌های دسته‌بندی‌شده نظیر `Education` و `Marital_Status` برای ساده‌سازی و بهبود کیفیت داده.
    * **حذف داده‌های پرت (Outlier Removal):** شناسایی و حذف نقاط داده‌ای غیرعادی (Outliers) با استفاده از روش Interquartile Range (IQR) برای افزایش کیفیت تحلیل.
    * **حذف ویژگی‌های نامربوط/تکراری:** حذف ستون‌هایی که یا تکراری هستند (اطلاعاتشان در ویژگی‌های جدید موجود است) یا برای تحلیل خوشه‌بندی مفید نیستند (مانند شناسه مشتری، یا ستون‌های با یک مقدار ثابت).
    * **استانداردسازی و انکدینگ:** اعمال مقیاس‌بندی (StandardScaler) بر ویژگی‌های عددی و انکدینگ ترتیبی (OrdinalEncoder) بر ویژگی‌های دسته‌بندی‌شده باقی‌مانده برای آماده‌سازی داده‌ها جهت مدل‌سازی.

* **کاهش ابعاد (Dimensionality Reduction):**
    * استفاده از **تحلیل مؤلفه‌های اصلی (PCA)** برای کاهش ابعاد مجموعه داده به 3 مؤلفه اصلی (`PC1`, `PC2`, `PC3`). این کار به بصری‌سازی آسان‌تر خوشه‌ها و افزایش کارایی الگوریتم‌های خوشه‌بندی کمک می‌کند.

* **خوشه‌بندی (Clustering):**
    * **پیاده‌سازی و ارزیابی:**
        * **KMeans:** پیاده‌سازی و ارزیابی عملکرد این الگوریتم.
        * **Agglomerative Clustering:** پیاده‌سازی و ارزیابی عملکرد این الگوریتم.
        * **Gaussian Mixture Models (GMM):** پیاده‌سازی و ارزیابی عملکرد این الگوریتم.
    * **تعیین تعداد بهینه خوشه‌ها:** استفاده از معیارهای متنوع برای یافتن بهترین تعداد خوشه‌ها:
        * **روش Elbow (آرنج) با WCSS:** برای KMeans.
        * **Silhouette Score:** برای ارزیابی چگونگی تفکیک خوشه‌ها.
        * **Davies-Bouldin Index:** معیاری برای ارزیابی فشرده‌سازی و تفکیک خوشه‌ها (مقادیر کمتر بهتر است).
        * **AIC (Akaike Information Criterion) و BIC (Bayesian Information Criterion):** برای GMM، که نشان‌دهنده تعادل بین پیچیدگی مدل و برازش آن هستند (مقادیر کمتر بهتر است).

* **تحلیل و بصری‌سازی خوشه‌ها (Cluster Analysis & Visualization):**
    * **مشخصات خوشه‌ها:** محاسبه میانگین و میانه ویژگی‌های کلیدی مانند درآمد، هزینه‌ها، و تعداد خریدهای آنلاین/حضوری برای هر خوشه.
    * **نمودارهای مقایسه‌ای:** بصری‌سازی میانگین درآمد و هزینه‌ها برای هر خوشه با استفاده از نمودارهای میله‌ای.
    * **نمودارهای سه‌بعدی تعاملی:** نمایش خوشه‌های شناسایی شده در فضای سه‌بعدی PCA با استفاده از Plotly Express، امکان مشاهده تعاملی خوشه‌ها و مراکز آن‌ها.
    * **تفسیر خوشه‌ها:** ارائه تفسیرهایی از الگوهای رفتاری و مالی هر خوشه (به عنوان مثال، "مشتریان پردرآمد کم‌خرج"، "خریداران متوازن"، "مصرف‌کنندگان زیاد" و "خریداران اقتصادی") بر اساس نتایج خوشه‌بندی.

## مجموعه داده

این پروژه از فایل `dataset.csv` استفاده می‌کند که حاوی اطلاعات مشتریان شامل مشخصات دموگرافیک، درآمد، تاریخ عضویت، و میزان هزینه‌ها در دسته‌های مختلف محصولات است.

## ساختار پروژه

```
customer_personality_clustering/
├── visualizations/                       # پوشه‌ای برای ذخیره نمودارها و بصری‌سازی‌های تولید شده
│   ├── bar_plot.png                      # نمونه‌ای از نمودار میله‌ای مقایسه‌ای خوشه‌ها
│   ├── ...                               # سایر تصاویر نمودارها مانند هیستوگرام‌ها، نمودار همبستگی، و نمودارهای ارزیابی خوشه‌بندی
├── customer_segmentation_clustering_en.ipynb  # نوت‌بوک Jupyter اصلی پروژه با توضیحات و کد به زبان انگلیسی
├── customer_segmentation_clustering_fa.ipynb  # نوت‌بوک Jupyter اصلی پروژه با توضیحات و کد به زبان فارسی
├── dataset.csv                           # مجموعه داده اصلی مورد استفاده در پروژه
└── requirements.txt                      # لیستی از تمام کتابخانه‌های پایتون مورد نیاز برای اجرای پروژه
```

## کتابخانه‌ها و فناوری‌های مورد استفاده

* **Python 3.x**
* `pandas`: برای دستکاری و تحلیل داده.
* `numpy`: برای عملیات عددی پیشرفته.
* `scikit-learn`: مجموعه‌ای جامع از ابزارهای یادگیری ماشین برای پیش‌پردازش، کاهش ابعاد (PCA) و الگوریتم‌های خوشه‌بندی (KMeans, AgglomerativeClustering, GaussianMixture).
* `matplotlib`: برای ایجاد نمودارهای ایستا.
* `seaborn`: برای بصری‌سازی‌های آماری جذاب و پیشرفته.
* `plotly.express`: برای ایجاد نمودارهای تعاملی، به‌ویژه نمودارهای سه‌بعدی خوشه‌ها.
* `warnings`: برای مدیریت پیام‌های هشدار.

## نحوه راه‌اندازی و اجرا

برای راه‌اندازی و اجرای این پروژه، مراحل زیر را دنبال کنید:

1.  **شبیه‌سازی مخزن (Clone the repository):**
    ابتدا مخزن پروژه را از گیت‌هاب به سیستم خود شبیه‌سازی کنید:
    ```bash
    git clone [https://github.com/sorna-fast/customer_personality_clustering.git](https://github.com/sorna-fast/customer_personality_clustering.git)
    cd customer_personality_clustering
    ```

2.  **ایجاد و فعال‌سازی محیط مجازی (Create and activate a virtual environment):**
    توصیه می‌شود برای مدیریت وابستگی‌ها یک محیط مجازی ایجاد کنید:

    * **برای کاربران ویندوز:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **برای کاربران macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **نصب وابستگی‌ها (Install dependencies):**
    پس از فعال‌سازی محیط مجازی، تمام کتابخانه‌های مورد نیاز پروژه را با استفاده از فایل `requirements.txt` نصب کنید:
    ```bash
    pip install -r requirements.txt
    ```

4.  **اجرای نوت‌بوک‌ها (Run the notebooks):**
    برای مشاهده و اجرای تحلیل‌ها و کدها، Jupyter Notebook یا Jupyter Lab را اجرا کنید:

    * **Jupyter Lab:**
        ```bash
        jupyter lab
        ```
    * **Jupyter Notebook:**
        ```bash
        jupyter notebook
        ```
    پس از اجرای یکی از دستورات بالا، یک صفحه در مرورگر شما باز می‌شود. می‌توانید نوت‌بوک‌های `customer_segmentation_clustering_en.ipynb` (نسخه انگلیسی) یا `customer_segmentation_clustering_fa.ipynb` (نسخه فارسی) را باز کرده و سلول‌ها را به ترتیب اجرا کنید تا نتایج تحلیل و بصری‌سازی‌ها را مشاهده کنید.

## نتایج و بینش‌های کلیدی

این پروژه با موفقیت چندین خوشه متمایز از مشتریان را شناسایی کرده است. هر خوشه دارای الگوهای رفتاری و مالی منحصربه‌فردی است که می‌تواند مبنای استراتژی‌های بازاریابی هدفمند قرار گیرد. برخی از دسته‌بندی‌های بالقوه مشتریان (با فرض تعداد بهینه خوشه‌ها و تفسیر نتایج):

* **High-Income Savers (مشتریان پردرآمد کم‌خرج):** مشتریانی با درآمد بالا که تمایل کمتری به هزینه‌کردن در محصولات مختلف دارند.
* **Balanced Shoppers (خریداران متوازن):** مشتریانی با درآمد متوسط و هزینه‌های متوازن در دسته‌های مختلف محصولات.
* **Big Spenders (مصرف‌کنندگان زیاد):** مشتریانی که در دسته‌های مختلف محصولات، به خصوص محصولات گران‌قیمت، هزینه‌های بالایی دارند.
* **Budget Shoppers (خریداران اقتصادی):** مشتریانی با درآمد پایین‌تر که هزینه‌های آن‌ها نیز به نسبت کمتر و در دسته‌های خاصی متمرکز است.

این بینش‌ها به کسب‌وکارها کمک می‌کند تا گروه‌های مشتریان خود را بهتر بشناسند، کمپین‌های بازاریابی شخصی‌سازی شده‌ای ایجاد کنند و تجربه مشتری را بهبود بخشند.

---

## مجوز
این پروژه تحت مجوز [MIT](LICENSE) منتشر شده است.



👋 امیدواریم این پروژه برای شما مفید باشد! 🚀

## ارتباط با توسعه‌دهنده  
    ایمیل: masudpythongit@gmail.com 
    تلگرام: https://t.me/Fast_programmer
🔗 حساب گیتهاب: [sorna-fast](https://github.com/sorna-fast)

---
```