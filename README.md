# HSIP Subjects Algorithm
These are the high-level steps in the algorithm.

1. Reads in dataset from "Sheet 1" in Excel file
2. Read in [rules.txt](#rules.txt) that contains a list of invalid entries
3. Removes rows from the dataset with at least two invalid entries or blank values
4. Removes rows that have an amount ≤ 0
5. **Name** column is split into 3 parts (regardless of actual names present): first, middle, last
5. Sets up rules for matching (see [Matching section](#Matching) below)
6. Performs matches via blocking
7. Assigns an `alexid` to each individual
8. Tabulate total amount for each indivudal
9. Write original dataset with two new columns (`alexid` and `total`) to xlsx

The source code can be found [here](#exploratory.py). There are other minor details in the code that I didn't mention. 

# Matching
Each of the following column has a score of 0 or 1 to indicate a match or not. Two rows are a match if they have a score ≥ 2.
- **Name**
- **Address**
- **SSN**

**Note:** The **Name** score is considered an average of the *first name* and *last name* score (middle name is not considered).

# Runtime
The PC can process about 7,000 pairs a second for matches. For a million pairs, that's about 2.5 minutes. The program will print out how many pairs it is computing.

# Matching Algorithm
We compare two strings using the [Jaro-Winkler distance](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance). A threshold of 0.85 is considered for a match.

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

