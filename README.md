# deformation
A deformation-based framework for learning solution mappings of PDEs defined on varying domains  

learner: https://github.com/jpzxshi/learner  
Environment-python: python=3.12.9 numpy=2.3.1 pytorch=2.6.0+cu126  
Environment-cpp: -std=c++17 eigen=3.4.0 libtorch=2.0.0+cu118 (Running codes by VS2022 on Windows 10)  

**Polygonal domains (D2E):**  
&nbsp;&nbsp;&nbsp;&nbsp;1. Run 'generate_poly_domains.py' for generating raw data.  
&nbsp;&nbsp;&nbsp;&nbsp;2. Run function 'poly_domain_d2e_data' in 'generate_training_data.py'  for integrating into training data.  
&nbsp;&nbsp;&nbsp;&nbsp;3. Run 'train_poly_domian_d2e.py' for training model.  

**Polygonal domains (D2D):**  
&nbsp;&nbsp;&nbsp;&nbsp;1. Run 'generate_poly_domains.py' for generating raw data.  
&nbsp;&nbsp;&nbsp;&nbsp;2. Run function 'poly_domain_d2d_data' in 'generate_training_data.py'  for integrating into training data.  
&nbsp;&nbsp;&nbsp;&nbsp;3. Run 'train_poly_domian_d2d.py' for training model.  

**Fully parameterized star domains (D2D):**  
&nbsp;&nbsp;&nbsp;&nbsp;1. Run 'generate_star_domains.py' for generating raw data.  
&nbsp;&nbsp;&nbsp;&nbsp;2. Run function 'star_domain_d2d_data' in 'generate_training_data.py'  for integrating into training data.  
&nbsp;&nbsp;&nbsp;&nbsp;3. Run 'train_star_domian_d2d.py' for training model.  

**Locally deformed square domains (D2E):**  
&nbsp;&nbsp;&nbsp;&nbsp;1. Run 'generate_square_domains.py' for generating raw data.  
&nbsp;&nbsp;&nbsp;&nbsp;2. Run function 'square_domain_d2e_data' in 'generate_training_data.py'  for integrating into training data.  
&nbsp;&nbsp;&nbsp;&nbsp;3. Run 'train_square_domian_d2e.py' for training model.  

**Annular domains (D2D):**  
&nbsp;&nbsp;&nbsp;&nbsp;1. Run 'generate_annular_domains.py' for generating raw data.  
&nbsp;&nbsp;&nbsp;&nbsp;2. Run function 'annular_domain_d2d_data' in 'generate_training_data.py'  for integrating into training data.  
&nbsp;&nbsp;&nbsp;&nbsp;3. Run 'train_annular_domian_d2d.py' for training model.  

**Hybrid Iterative Method:**  
&nbsp;&nbsp;&nbsp;&nbsp;1. Run 'HIM/cpp/main.cpp' for performing hybrid iterative method.