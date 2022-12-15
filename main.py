import csv
from PyPDF2 import PdfReader
from sklearn import tree

class Specification:
  def __init__(self, pdf_file_path):
    self.yToFeatures = {}
    self.pdf_file_path = pdf_file_path
    self.get_features()

  def get_features(self):
    reader = PdfReader("/Users/jeehwancho/repos/pdf_extractor/assets/PM1-815a-single.pdf")
    reader = PdfReader(self.pdf_file_path)
    page_number = -1
    def visitor_body(text, cm, tm, fontDict, fontSize):
      x = tm[4]
      y = tm[5]
      if text == "" or text == "\n":
        return
      if y in self.yToFeatures:
        self.yToFeatures[y]['text'] += text
      else:
        self.yToFeatures[y] = {
          'x': x,
          'y': y,
          'page_number': page_number,
          'text': text,
        }
    for i, page in enumerate(reader.pages):
      page_number = i + 1
      page.extract_text(visitor_text=visitor_body)
    self.clean()

  def export_to_csv(self, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
      w = csv.DictWriter(csvfile, delimiter='\t', fieldnames=['Y', 'x', 'y', 'text', 'page_number'])
      w.writeheader()
      for key in self.yToFeatures:
        features = self.yToFeatures[key]
        w.writerow({
          'Y': '0',
          'x': features['x'],
          'y': features['y'],
          'text': features['text'],
          'page_number': features['page_number'],
          })

  def import_csv(self, csv_file_path, feature_keys):
    X = []
    Y = []
    with open(csv_file_path, newline='') as csvfile:
      r = csv.DictReader(csvfile, delimiter='\t')
      for row in r:
        Y.append(row['Y'])
        features = []
        for feature_key in feature_keys:
          features.append(row[feature_key])
        X.append(features)
    return X, Y

  def clean(self):
    keysToBeDeleted = []
    for key in self.yToFeatures:
      features = self.yToFeatures[key]
      features['text'] = features['text'].strip()
      if features['text'] == "":
        keysToBeDeleted.append(key)
      else:
        self.yToFeatures[key] = features
    for key in keysToBeDeleted:
      del self.yToFeatures[key]

pdf_file_path = "/Users/jeehwancho/repos/pdf_extractor/assets/PM1-815a-single.pdf"
csv_file_path = "data1.csv"
spec = Specification(pdf_file_path)
spec.export_to_csv(csv_file_path)
X, Y = spec.import_csv(csv_file_path, ['x', 'y', 'text'])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
res = clf.predict([[2., 2.]])
print(res)

# Scikit-learn: https://scikit-learn.org/stable/modules/tree.html
# X: [[x, y, text]]
# Y: 0 = nothing, 1 = beginning of section, 2 = end of section

# TODO:
# (0) git init
# (0) use page.mediabox to get relative x, y
# (1) generate csv for PM1-815a-single.pdf
# (2) generate csv for PM1-815a.pdf, and feed X, evaluate Y
