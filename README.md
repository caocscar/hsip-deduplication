# Assumptions

These are assumed to be invalid entries:
1. address = `1111 MAIN ST` or null
2. city = `ANYWHERE` or null
3. ssn = `111111111` or null

Eliminating rows with 2 or more invalid entries.

# Matching
Two rows are a match if they match on the following two identifiers:
- SSN + Name
- SSN + Address
- Name + Address


