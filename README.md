# Small-or-Family-Business-Project
In this project we evaluate the gap in the management literature between small businesses and family businesses

This is a python script that essentially loads data from an Excel file containing management academic papers metadata and generates Iramuteq tags from these metadata.

In addition when papers are duplicate but classified in the same theme (small or family business), they are deleted. If there is a duplicate but each instance classified in a different theme, we use the OpenAI API to check whether the paper should be classified as small or family business, and the incorrect instance is deleted.

Two outputs are offered - 1 an Excel file with the new columns, and 2 - an UTF-8 .txt file in the appropriate Iramuteq format containing the tags generated.
