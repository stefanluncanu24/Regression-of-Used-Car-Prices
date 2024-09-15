<!DOCTYPE html>
<html>
    
<h1>Code Overview</h1>

<p>This repository contains code for data preprocessing, feature engineering, and building a machine learning model to predict car prices. The project involves cleaning and transforming data, extracting features, and using LightGBM with hyperparameter optimization to build a predictive model.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#feature-engineering">Feature Engineering</a></li>
    <li><a href="#model-building">Model Building</a></li>
    <li><a href="#hyperparameter-tuning">Hyperparameter Tuning</a></li>
    <li><a href="#model-evaluation">Model Evaluation</a></li>
    <li><a href="#predictions">Predictions</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
</ul>

<h2 id="data-preprocessing">Data Preprocessing</h2>

<h3>Loading and Merging Data</h3>

<ul>
    <li><strong>Loading Data</strong>: The training and test datasets are loaded from <code>train.csv</code> and <code>test.csv</code>.</li>
    <li><strong>Merging Data</strong>: Both datasets are concatenated into a single DataFrame <code>dataset</code> for uniform preprocessing.</li>
</ul>

<pre><code class="language-python">train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
dataset = pd.concat([train, test], axis=0).reset_index(drop=True)
</code></pre>

<h3>Data Cleaning</h3>

<p>The <code>clean_data</code> function performs several data cleaning steps:</p>

<ul>
    <li><strong>Car Age Calculation</strong>: Derives a new column <code>CarAge</code> by subtracting <code>model_year</code> from 2024.</li>
    <li><strong>Engine Feature Extraction</strong>: Extracts <code>HP</code> (horsepower), <code>L</code> (engine capacity in liters), and <code>Cylinder</code> counts from the <code>engine</code> column using regular expressions.</li>
    <li><strong>Electric Vehicle Handling</strong>:
        <ul>
            <li>Fills <code>NaN</code> values in <code>HP</code>, <code>L</code>, <code>Cylinder</code>, and <code>fuel_type</code> with <code>0</code> for electric vehicles.</li>
            <li>Replaces missing <code>fuel_type</code> values with <code>'Electric'</code>.</li>
        </ul>
    </li>
    <li><strong>Missing Value Imputation</strong>:
        <ul>
            <li>Replaces missing <code>HP</code> values with the median.</li>
            <li>Replaces missing <code>L</code> and <code>Cylinder</code> values with the mean (rounded).</li>
            <li>Fills missing <code>fuel_type</code> values with the mode of the same model or overall mode.</li>
            <li>Replaces missing <code>accident</code> and <code>clean_title</code> values with <code>'missing'</code>.</li>
        </ul>
    </li>
    <li><strong>Column Dropping</strong>: Removes unnecessary columns like <code>model_year</code>, <code>engine</code>, and <code>id</code>.</li>
</ul>

<pre><code class="language-python">def clean_data(df):
    # Car Age Calculation
    df.insert(4, "CarAge", 2024 - df["model_year"])
    df = df.drop(columns=['model_year'])

    # Engine Feature Extraction
    df.insert(5, 'HP', df['engine'].apply(lambda x: re.search(r'(\d+\.?\d*)\s*HP', x).group(1) if re.search(r'(\d+\.?\d*)\s*HP', x) else None))
    df.insert(6, 'L', df['engine'].apply(lambda x: re.search(r'(\d+\.?\d*)\s*L', x).group(1) if re.search(r'(\d+\.?\d*)\s*L', x) else None))
    df.insert(7, 'Cylinder', df['engine'].apply(lambda x: (
        re.search(r'(\d+)\s*Cylinder', x).group(1) if re.search(r'(\d+)\s*Cylinder', x) 
        else (re.search(r'V(\d+)', x).group(1) if re.search(r'V(\d+)', x) else None))
    ))

    # Convert to Numeric Types
    df['HP'] = pd.to_numeric(df['HP'], errors='coerce')
    df['L'] = pd.to_numeric(df['L'], errors='coerce')
    df['Cylinder'] = pd.to_numeric(df['Cylinder'], errors='coerce')

    # Electric Vehicle Handling
    def fillna_for_electric(df):
        electric_condition = df['engine'].str.contains('Electric', case=False, na=False)
        df.loc[electric_condition, ['HP', 'L', 'Cylinder', 'fuel_type']] = df.loc[electric_condition, ['HP', 'L', 'Cylinder', 'fuel_type']].fillna(0)
        return df
    df = fillna_for_electric(df)
    df = df.drop(columns=['engine'])
    df['fuel_type'] = df['fuel_type'].replace(0, 'Electric')

    # Drop 'id' Column
    df = df.drop(columns=['id'])

    # Missing Value Imputation
    df = df.fillna({'HP': df['HP'].median()})
    df = df.fillna({'L': df['L'].mean().round()})
    df = df.fillna({'Cylinder': df['Cylinder'].mean().round()})
    overall_mode = df['fuel_type'].mode()[0]
    df['fuel_type'] = df.groupby('model')['fuel_type'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else overall_mode))
    df = df.fillna({'accident': "missing"})
    df = df.fillna({'clean_title': "missing"})

    return df

df = clean_data(dataset)
</code></pre>

<h2 id="feature-engineering">Feature Engineering</h2>

<h3>Transmission Standardization</h3>

<ul>
    <li><strong>Standardization</strong>: Standardizes the <code>transmission</code> column by mapping various transmission types to standardized labels (e.g., <code>'Automatic'</code>, <code>'Manual'</code>, <code>'CVT'</code>).</li>
    <li><strong>Handling Missing Values</strong>: Assigns <code>'Unknown'</code> to missing or unknown transmission types.</li>
</ul>

<pre><code class="language-python"># Replace en-dash with NaN
df['transmission'] = df['transmission'].replace("–", np.nan)

# Transmission Mapping
transmission_mapping = {
    "a/t": "Automatic",
    # ... (other mappings)
    "dct": "Dual-Clutch",
}

# Map Known Transmissions
df['transm_'] = df['transmission'].map(transmission_mapping)

# Define Mapping Function
def map_transmission(trans):
    trans = str(trans).lower()
    # ... (mapping logic)
    return 'Unknown'

# Apply Mapping to Unmapped Entries
mask = df['transm_'].isnull()
df.loc[mask, 'transm_'] = df.loc[mask, 'transmission'].apply(map_transmission)
df['transm_'].fillna('Unknown', inplace=True)
</code></pre>

<h3>Color Standardization</h3>

<ul>
    <li><strong>Standardization</strong>: Standardizes interior (<code>int_col</code>) and exterior (<code>ext_col</code>) color columns.</li>
    <li><strong>Base Color Extraction</strong>: Extracts base colors from color descriptions using predefined dictionaries.</li>
</ul>

<pre><code class="language-python"># Define Color Replacement Dictionaries
int_replacements = {
    'Medium Earth Gray': 'Gray',
    # ... (other mappings)
    'WHITE': 'White'
}
ext_replacements = {
    'Blu': 'Blue',
    # ... (other mappings)
    'Caviar': 'Black'
}

# Standardize and Extract Base Colors
def standardize_and_extract_colors(df, int_replacements, ext_replacements):
    df['int_col'] = df['int_col'].replace(int_replacements)
    df['ext_col'] = df['ext_col'].replace(ext_replacements)
    df['int_col'] = df['int_col'].str.lower()
    df['ext_col'] = df['ext_col'].str.lower()
    # ... (base color extraction logic)
    return df

df = standardize_and_extract_colors(df, int_replacements, ext_replacements)
</code></pre>

<h3>Additional Feature Extraction</h3>

<ul>
    <li><strong>Luxury Brand Indicator</strong>: Creates an <code>Is_Luxury_Brand</code> column to indicate if the car brand is considered luxury.</li>
    <li><strong>Mileage Features</strong>:
        <ul>
            <li>Calculates <code>Mileage_per_Year</code> by dividing <code>milage</code> by <code>CarAge</code>, handling division by zero.</li>
            <li>Computes <code>milage_with_age</code> and <code>Mileage_per_Year_with_age</code> as mean values grouped by <code>CarAge</code>.</li>
        </ul>
    </li>
</ul>

<pre><code class="language-python">def extract_other_features(df):
    # Luxury Brand Indicator
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                     'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                     'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
    df.drop(columns=['transmission'], inplace=True)

    # Mileage Features
    df['Mileage_per_Year'] = np.where(
        df['CarAge'] == 0,
        df['milage'],
        df['milage'] / df['CarAge']
    )
    df['milage_with_age'] = df.groupby('CarAge')['milage'].transform('mean')
    df['Mileage_per_Year_with_age'] = df.groupby('CarAge')['Mileage_per_Year'].transform('mean')

    return df

df = extract_other_features(df)
</code></pre>

<h2 id="model-building">Model Building</h2>

<h3>Data Splitting</h3>

<ul>
    <li><strong>Training and Test Sets</strong>: Splits the preprocessed data back into training and test sets based on the presence of the <code>price</code> column.</li>
    <li><strong>Validation Set</strong>: Further splits the training data into training and validation sets.</li>
</ul>

<pre><code class="language-python">train = df[df['price'].notnull()]
test = df[df['price'].isnull()]

X = train.drop(columns=['price'])
y = train['price']

# Split into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)
</code></pre>

<h3>Preprocessing for Modeling</h3>

<ul>
    <li><strong>Identifying Columns</strong>: Identifies categorical and numerical columns.</li>
    <li><strong>Preprocessing Pipeline</strong>: Uses a <code>ColumnTransformer</code> and <code>OneHotEncoder</code> to preprocess categorical features.</li>
    <li><strong>Pipeline Creation</strong>: Defines a pipeline to streamline preprocessing and model training.</li>
</ul>

<pre><code class="language-python">categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ],
    remainder='passthrough'
)

def create_pipeline(model):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline
</code></pre>

<h2 id="hyperparameter-tuning">Hyperparameter Tuning</h2>

<ul>
    <li><strong>Search Space Definition</strong>: Defines a search space for LightGBM hyperparameters, including <code>num_leaves</code>, <code>learning_rate</code>, <code>feature_fraction</code>, and <code>num_boost_round</code>.</li>
    <li><strong>Objective Function</strong>: Implements an <code>objective</code> function to minimize the root mean squared error (RMSE) using cross-validation.</li>
    <li><strong>Optimization Execution</strong>: Runs the optimization for a specified number of evaluations to find the best hyperparameters.</li>
</ul>

<pre><code class="language-python">lgbm_space = {
    'num_boost_round': scope.int(hp.quniform('num_boost_round', 100, 2500, 1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 20, 600, 1)),
    'learning_rate': hp.loguniform('learning_rate', -9.21, -1.01),
    'feature_fraction': hp.uniform('feature_fraction', 0.3, 1),
    'random_state': rs,
}

def objective(params, model_type='lgbm'):
    # ... (objective function code)
    return {'loss': rmse, 'status': STATUS_OK}

def optimize(space, model_type, max_evals=50):
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, model_type=model_type),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(rs)
    )
    return best, trials

# Run Optimization
best_lgbm_params, trials_lgbm = optimize(lgbm_space, 'lgbm', max_evals=50)
</code></pre>

<h2 id="model-evaluation">Model Evaluation</h2>

<ul>
    <li><strong>Model Training</strong>: Trains a LightGBM model using the optimized hyperparameters.</li>
    <li><strong>Validation</strong>: Evaluates the model on the validation set and computes the RMSE.</li>
    <li><strong>Performance Assessment</strong>: Prints the RMSE to assess model performance.</li>
</ul>

<pre><code class="language-python">best_lgbm = lgb.LGBMRegressor(
    num_leaves=int(best_lgbm_params['num_leaves']),
    learning_rate=best_lgbm_params['learning_rate'],
    feature_fraction=best_lgbm_params['feature_fraction'],
    random_state=rs,
    n_estimators=int(best_lgbm_params['num_boost_round'])
)

pipeline_lgbm = create_pipeline(best_lgbm)
pipeline_lgbm.fit(X_train, y_train)
preds_lgbm = pipeline_lgbm.predict(X_val)
rmse_lgbm = np.sqrt(mean_squared_error(y_val, preds_lgbm))
print(f"\nLightGBM RMSE on validation set: {rmse_lgbm}")
</code></pre>

<h2 id="predictions">Predictions</h2>

<ul>
    <li><strong>Final Model Training</strong>: Fits the final model on the entire training data.</li>
    <li><strong>Test Set Predictions</strong>: Generates predictions on the test set.</li>
    <li><strong>Exporting Results</strong>: Exports the predictions to <code>submission.csv</code>.</li>
</ul>

<pre><code class="language-python"># Fit on Entire Training Data
pipeline_lgbm.fit(X, y)

# Predictions on Test Set
X_test = test.drop(columns=['price'])
preds_test_lgbm = pipeline_lgbm.predict(X_test)

# Export Predictions
output = pd.DataFrame({'id': test.index, 'price': preds_test_lgbm})
output.to_csv('submission.csv', index=False)
</code></pre>

<h2 id="usage">Usage</h2>

<p>To run the code:</p>

<ol>
    <li><strong>Install Dependencies</strong>: Ensure all dependencies are installed.</li>
    <li><strong>Data Placement</strong>: Place <code>train.csv</code> and <code>test.csv</code> in the working directory.</li>
    <li><strong>Execute Script</strong>: Run the script to perform data preprocessing, model training, and prediction generation.</li>
</ol>

<h2 id="dependencies">Dependencies</h2>

<ul>
    <li><strong>Python 3.x</strong></li>
    <li><strong>Pandas</strong></li>
    <li><strong>NumPy</strong></li>
    <li><strong>LightGBM</strong></li>
    <li><strong>scikit-learn</strong></li>
    <li><strong>Hyperopt</strong></li>
    <li><strong>Regular Expressions (<code>re</code> module)</strong></li>
</ul>

</body>
</html>
