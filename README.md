# HSIP Person Identification
CSCAR's script for HSIP person identification.

## Usage
To run the program, do the following:
1. Open `Anaconda Prompt` from the Start Menu
2. Change directory to the folder where the python script `hsip_alexid.py` and **Excel** file resides.  
For example, `cd C:\Users\caoa\Desktop\HSIP`
3. Run the program with some input arguments.  
For example, `python hsip_assignid.py --filename "my file.xlsx"`
4. The program will create an output file by append `_alexid` to the end of the filename.  
The output file will have two extra columns: **alexid** and **total** for that individual.

## Example
Here is a sample command line which uses all the available keyword arguments.  
`python hsip_alexid.py --filename "my file.xlsx" --threshold 0.85`

Argument|Shorthand|Usage
---|:---:|---
--filename|-f|`python hsip_alexid.py -f input_file.xlsx`
--threshold|-t|`python hsip_alexid.py --filename input_filet.xlsx --threshold 0.9`

**Tip:** You have to specify a `--filename` argument or it will complain.

## Runtime
The PC can process about 7,500 pairs a second for matches. For a million pairs, that's about 2 minutes. The program will print out how many pairs it is computing.

## Algorithm
These are the high-level steps in the algorithm.
1. Reads in dataset from "Sheet 1" in Excel file
2. Read in [rules.txt](rules.txt) that contains a list of invalid entries
3. Removes rows from the dataset with at least two invalid entries or blank values among **name**, **ssn**, **address**
4. Removes rows that have an amount ≤ 0
5. Convert address to lowercase and remove spaces for matching
6. Convert name to lowercase for matching
7. **Name** column is split into 3 parts (regardless of actual names present): first, middle, last
8. Sets up rules for matching (see [Matching section](#matching) below)
9. Performs matches using blocking
10. Assigns an `alexid` to each individual
11. Tabulate total amount for each indivudal
12. Write original dataset with two new columns (`alexid` and `total`) to xlsx

The source code can be found [here](hsip_alexid.py). There are other minor details in the code that I didn't mention. 

## Valid and Invalid Street Addresses
Three parts of the street address: *address_1, city, postal* are considered for the street address filtering.
An address is considered valid if the **address score** is ≥ 1 otherwise it is invalid. The weights were chosen such that a valid score meant that the mailing address could be derived from the relevant entries. The following table show the relative weights.

Part|Score
---|:---:
address|0.6
city|0.4
postal|0.4

The following table shows the exhaustive examples from these three columns.

Valid Parts|Example|Address Score|Street Address
---|---|:---:|:---:
address + city + postal|1107 White St, Ann Arbor, 48103|1.4|Valid
address + city|1107 White St, Ann Arbor|1.0|Valid
address + postal|1107 White St, 48104|1.0|Valid
city + postal|Ann Arbor, 48104|0.8|Invalid
address|1107 White St|0.6|Invalid
city|Ann Arbor|0.4|Invalid
postal|48104|0.4|Invalid

## Matching
Each of the following column has a score to indicate a match or not. A score ≥ 1 indicates a match for that column. Two rows are a match if they have a score ≥ 2.

Column|Max Score Possible
---|:---:
Name|1.0
Street Address|1.4
SSN|1.0

**Note:** The **Name** score is considered an average of the *first name* and *last name* score (middle name is not considered).

## Matching Algorithm
We compare two strings using the [Jaro-Winkler distance](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance). A default threshold of 0.85 is considered for a match (i.e. score = 1)

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

