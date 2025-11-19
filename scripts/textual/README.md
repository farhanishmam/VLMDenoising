The dataset we are using is <a href="https://huggingface.co/datasets/cambridgeltl/DARE"> DARE Dataset </a> (Sterz et al)

1.  <b> Download the Dataset </b>
To download the dataset (without the heavy img column for faster processing), run:

    ```
    python download_dataset.py
    ```
    
    This will generate the file:
    ```
    data/[without images]1_correct_validation.csv 
    ```
2. <b> Add Noise to Questions </b>
To introduce noise into the `question` column of the `data/[without images]1_correct_validation.csv` CSV, just run the below command: 

    ```
    python noise_addition.py
    ```
    
    This will produce:
    ```
    data/NoisyQuestionPairs.csv
    ```

    The new file will include two additional columns:

    - `modified_question`

    - `modified_question_function_name`

3. <b> Add Denoised Questions </b>
To create denoised versions of the noisy questions stored in the `modified_question` column of `data/NoisyQuestionPairs.csv`, run:

    ```
    python denoise_script.py
    ```
    
    This will produce the <b>final CSV</b>:
    ```
    data/Noisy-Denoised_QuestionPairs.csv
    ```
    The new file will include another additional columns:

    - `denoised_question`

    the first row can be looked like below:

    ```csv
    id,instance_id,question,answer,A,B,C,D,category,path,modified_question,modified_question_function_name,denoised_question
    vcr_2321,2321,what are they doing,what are they doiÉ´g,What are they doing?,substitute_with_homoglyphs,['C'],they are discussing divorce,they are sheltering from the rain,they are on holiday and enjoying a break from walking,they are waiting on a bus,vcr,000000130826.jpg
    ```