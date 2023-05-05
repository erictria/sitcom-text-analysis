# Sitcom Text Analysis

## Metadata
- Project: Sitcom Text Analysis
- Class: DS 5001 Spring 2023
- Version Date: May 5, 2023
- Author: Eric Tria
- Email: ericmtria@gmail.com / emt4wf@virginia.edu

## Synopsis

This project conducts exploratory text analytics on episode scripts from three famous sitcoms:
- Parks and Recreation
- Brooklyn Nine-Nine
- The Office US

## Data

Episode scripts are scraped from [Sublikescript.com](https://subslikescript.com/). Additional details on the episodes and seasons are accessed from Wikipedia.

## Important Files

#### Notebooks

- `FINAL_REPORT.ipynb` - summarizes the entire process of the project
- `EXTRACT_DATA.ipynb` - performs the data scraping and acquisition
- `TEXT_ANALYSIS.ipynb` - performs the code for model building and analysis

#### Python Files

- `corpus_enhancer.py` - functions for adding features to CORPUS and VOCAB tables
- `script_scraper.py` - functions for extracting data from Sublikescript.com
- `text_helper.py` - functions for conducting analysis done on TEXT_ANALYSIS.ipynb
- `wiki_scraper.py` - functions for extracting data from Wikipedia

## Manifest
```
sitcom-text-analysis/
    lib/
        corpus_enhancer.py
        script_scraper.py
        text_helper.py
        wiki_scraper.py
    pictures/
        ...
    FINAL_REPORT.ipynb
    EXTRACT_DATA.ipynb
    TEXT_ANALYSIS.ipynb
    LICENSE
    README.md
    .gitignore
```