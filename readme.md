Scope out the pdf ml proj

Features per text: x, y, page, delta x, part (header, footer, body), row in part, etc (possibly add prev/next relations)

Want: section starts and ends

TODO:
- pdf reader to normalize data
- find ML/AI library to use
- find pdfs to test it out (in pdf_exporter_service)
- implement get_features(text)
- write sample answers to train
- start with 1/2 x-validation
- improve

source .venv/bin/activate
python3 -m pip install matplotlib