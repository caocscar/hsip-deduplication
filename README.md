# HSIP Person Identification
CSCAR's script for HSIP person identification.

## Installation
1. Download the Anaconda Python package installer https://repo.continuum.io/archive/Anaconda3-5.2.0-Windows-x86_64.exe. This downloads Anaconda Python 3.6 for Windows (631.3MB) from the repo archive.
2. Install the package by clicking on the executable `Anaconda3-5.2.0-Windows-x86_64.exe`.
3. Choose to install it just for the current user.
4. After installation is done, we need to still install some additional packages. Open `Anaconda Prompt` from the Start Menu.
5. Run the following commands. They will install the required dependencies that are not part of the default Anaconda installation.  
`pip install recordlinkage`  
`pip install nameparser`
6. Download zip of [github repository](https://github.com/caocscar/hsip-deduplication)
or git clone `git clone https://github.com/caocscar/hsip-deduplication.git`
7. Unzip file to working directory (if necessary).
8. Put input file in the same directory as `hsip_rollupid.py`.

## Usage
1. Open `Anaconda Prompt` from the Start Menu
2. Change directory to the folder where the python script `hsip_rollupid.py` and **Excel** file resides.  
For example, `cd C:\Users\caoa\Desktop\HSIP`
3. Run the program with some input arguments.  
For example, `python hsip_rollupid.py --filename "my file.xlsx"`
4. The program will create an output file by append `_rollupid` to the end of the filename.  

## Example
Here is a sample command line which uses all the available keyword arguments.  
`python hsip_rollupid.py --filename "my file.xlsx"`

Argument|Shorthand|Usage
---|:---:|---
--filename|-f|`python hsip_rollupid.py -f input_file.xlsx`

**Tip:** You have to specify a `--filename` argument or it will complain.

## Input File Requirements
1. Any password protection should be removed from the file prior to running the program otherwise it will result in an error.
2. The file should have at most two sheets. The first sheet should have the input data 
The following table specifies the possible columns along with whether it is required and if the name is case sensitive:

---|hsip|sid|hsip_sid|HUM #|name|email|ssn|address_1|address_2|city|county|state|postal|method|date|amt|entered|updated|status|origname|origemail|origssn|origaddress|AP Control|new_rollupid|TIN MATCH|NOTES
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
required|yes|no|no|no|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes
name case-sensitive|yes|no|no|no|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|yes|no|no|no|no|yes|yes|yes|yes

## Output File
The program will add/modify the following columns in the output file.

column|
---|
rollupid|
total_rollup|
name_ct|
email_ct|
ssn_ct|
address_ct|
ct_sum|
rollup_rank|
same_ssn_diff_rollupid|
same_name_diff_rollupid|
same_address1_diff_rollupid|
same_email_diff_rollupid|

## Algorithm
These are the high-level steps in the algorithm.
1. Reads in dataset from a single Excel file.
2. Standardize SSN
    - fill blanks with 000000000
    - remove any hyphens
    - replace 111111111 with 000000000
    - add necessary leading zeros to make a 9 digit string
3. Standardize Name
    - move any email addresses to email column
    - keep only valid punctuation `-&` and remove the rest
    - remove the following titles *(MD,PHD,FCCP,DDS,MBA,MHS**
    - convert to UPPERCASE
4. Standardize Address (address_1 and address_2)
    - swap **address_1** and **address_2** if **address_1** is blank and **address_2** is not
    - swap **address_1** and **address_2** if there is a a `C/O` in **address_1**
    - combine **address_1** and **address_2** if there is only a number in **address_1**
    - move any email addresses to email column
    - keep only valid punctuation `-#/` and remove the rest
    - convert to UPPERCASE
5. Standardize Email
    - convert to lowercase
    - remove punctuation `._`
6. Read in [rules.txt](rules.txt) that contains a list of invalid entries.
7. Removes rows from the dataset with at least two invalid entries or blank values among **name**, **ssn**, **address**, **email**.
8. Add invalid rows to the `invalid_rows` sheet.

9. Standardize common address words (e.g. Street to St) and direction (e.g. West to W).
10. Remove address spaces and commas for faster matching.

11. Grab local part (characters before the @) of email address.

12. Replace hyphens with spaces for name for better matches.
13. Parse name into `first` and `last` using the nameparser package.
14. Created **initials** column from **first** and **last** name.

15. Define rules for matching (see [Matching section](#matching) below).

16. Performs five rounds of matching using these blocking elements:
    - **ssn**
    - **email**
    - **address**
    - **last** & **initials**
    - **first** & **initials**

17. Tabulate total score for each row that was compared
18. Assigns a match to those pair with a score ≥ 2
19. Creates a network graph and assigns a **rollupid** to each individual using the network's connected components.
20. Override any **rollupid** with a manually assigned **new_rollupid**

21. Calculate the total amount for each **rollupid**
22. Get distinct count of name, email, ssn, address for each **rollupid**. A blank counts as a distinct value. 
23. Enumerate records within each **rollupid** based on **date** column in reverse chronological order.

24. Format datetime columns **date** and **entered**  to only `Month-Day-Year`

25. Create new columns to identify possible false negatives for further manual inspection  
`same_ssn_diff_rollupid`  
`same_email_diff_rollupid`
`same_name_diff_rollupid`  
`same_address1_diff_rollupid`  
The algorithm will flag rows that have the same {ssn, name, address, email} but   different rollupid (a grouping). Certain rows will not be flagged if they meet the one of the following conditions:
    - A row that is a duplicate of another based on these 4 columns. This is based on the input columns to the matching process. For example, *1128 South White Avenue* and *1128 S White Ave* will be considered identical.
    - If all the values in the grouping are `000000000`
    - If the value is an invalid entry as listed in `rules.txt`
    - For address, any value that is considered common (found in more than 5 rows). This is to reduce the number of records flagged to look at. Addresses that are common are usually a university/business location. For example, *1500 E Medical Ctr Dr* is UM Hospital and *1285 Franz Hall Box 951563* is the UCLA Pyschology Dept.

26. Write original dataset with new columns.
27. Write additional sheet containing **invalid_rows**.

The source code can be found [here](hsip_rollupid.py). There are other minor details in the code that I didn't mention. 

## Matching
Each of the following column has a score to indicate a match or not. A non-zero score indicates a match for that column. Two rows are a match if they have a score ≥ 2.

Column|Score
---|:---:
first|0.5
last|0.5
email|1.5
ssn|1.0
address|1.0

**Note:** If there is only one name, the algorithm considers it a **first** name.

## String Matching
We compare two strings using the [Jaro-Winkler distance](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance). A threshold of 0.82 is considered for a match.

The table below shows some example of some string comparisons.

String 1|String 2|Jaro-Winkler Distance
:---:|---|:---:
alex|alexis|0.93
alex|alexander|0.89
alex|alec|0.88
alex|blex|0.83
alex|alan|0.67
alex|lily|0.50
alex|zhou|0.00

## SSN Matching
We compare two SSN strings using the [Demarau-Levenshtein distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance). A threshold of 0.77 is considered for a match.

## Runtime
The program will print out occasional messages indicating progress. Running time will vary on PC hardware. It should take less than 10 minutes for a file with approximately 200,000 rows. On my Intel i5-4590 3.30GHz with 8GB RAM, it took about 7 minutes. 

If it takes more than 15 minutes, there is probably something wrong and should be diagnosed.

## Progress Print Out
The print out should look something like this if it runs without error.

Reading excel file 20190314_Working_March_rollupid.xlsx
This section took 42 seconds
Standardizing ssn
Standardizing name
Standardizing address
Standardizing email
Reading rules.txt file
Filtering out invalid rows
Preparing address, email, name columns for matching purposes
84 seconds have elapsed already
Starting Matching Process

ssn pairs = 70,959
Matching took 4.5 seconds
Pairs per second: 15867

email_ pairs = 48,758
Matching took 1.7 seconds
Pairs per second: 28299

address_ pairs = 2,976,469
Matching took 83.7 seconds
Pairs per second: 35543

['last', 'initials'] pairs = 950,536
Matching took 36.7 seconds
Pairs per second: 25886

['first', 'initials'] pairs = 2,082,191
Matching took 75.1 seconds
Pairs per second: 27708

Assigning rollupid to rows
Replaced 33 rows with manual rollupid
Calculating _ct columns and total_rollup

Identifying possible false negatives
5 same ssn have different rollupids
926 same names have different rollupids
1450 same addresses have different rollupids
1 same emails have different rollupids
Creating output excel file 20190314_Working_March_rollupid_processed.xlsx
301 seconds have elapsed already
20190314_Working_March_rollupidv2.xlsx created in 122 seconds
This whole process took too long: 423 seconds
