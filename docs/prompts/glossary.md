# Glossary Generation

## Prompt

!!! prompt
    Please generate an glossary of terms for the 250 most frequently used terms used in an introductory course 
    in introduction to data science with Python.
    Make sure to include the names of Python libraries used in data science.
    The target audience for this glossary is college freshmen.
    Each term should have a term name in a level-4 markdown (####) and the
    definition placed in the body text.
    Do not use the term in the definition of the term.
    The definition should be precise, concise and distinct.
    
    If appropriate, create an **Example:** of how that term is used in the
    Introduction to Data Science with Python course.  Do not place a newline after the Example:
    Return the glossary of terms in alphabetical order.

    A term definition is considered to be consistent with ISO metadata registry guideline 11179 if it meets the following criteria:

    1. Precise
    2. Concise
    3. Distinct
    4. Non-circular
    5. Unencumbered with business rules

## Update

We have generated a lot of new content in the @docs/chapters area.  However, our @docs/glossary.md might not include all 
the new terms that it should include.  Go through each of the new chapters and create a list of terms that an average 
high-school student might not understand.  Then check if these terms are already in the glossary.  If they are not in the 
glossary, add the term and create clear definitions for each term.  Give an example if appropriate.  Use the existing terms 
as examples. 

