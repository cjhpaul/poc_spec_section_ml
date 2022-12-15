import os
import csv
from PyPDF2 import PdfReader
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import graphviz

# features based on: https://fieldwire.atlassian.net/wiki/spaces/ENGINEERIN/pages/1298890843/Specification+File+Automation+-+section+detection
# features_dictionary[{page_num}-{y}] = {'x': 123, 'y': 321, 'text': 'foobar', 'page_number': 123}
# features have to be float
class Specification:
  # advanced features: diff_prev_x, diff_next_x
  feature_keys = ['page', 'rel_x', 'rel_y', 'text', 'len', 'section_idx', 'end_idx', 'has_6_digits', 'x', 'y']
  default_Y_value = "none"

  def __init__(self, feature_keys_to_use):
    self.feature_keys_to_use = feature_keys_to_use
    self.clf_rf = RandomForestClassifier(n_estimators = 100, max_features = "sqrt", class_weight = {'none': 1, 'ss': 10, 'se': 10})
    self.clf_dt = DecisionTreeClassifier(max_features = "sqrt", class_weight = {'none': 1, 'ss': 10, 'se': 10})

  def train(self, training_csv_folder):
    filtered_X, Y = Specification.get_XY_from_folder(training_csv_folder, self.feature_keys_to_use)
    self.clf_rf = self.clf_rf.fit(filtered_X, Y)
    self.clf_dt = self.clf_dt.fit(filtered_X, Y)

  def score(self, training_csv_folder):
    X, Y = Specification.get_XY_from_folder(training_csv_folder, self.feature_keys_to_use)
    cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    # scores = cross_val_score(self.clf, X, Y, cv=5, scoring = 'f1_macro')
    scores = cross_val_score(self.clf_rf, X, Y, cv=cv)
    print(f'Random Forest: {round(sum(scores)/len(scores)*100, 3)}')
    print(scores)
    scores = cross_val_score(self.clf_dt, X, Y, cv=cv)
    print(f'Decision Tree: {round(sum(scores)/len(scores)*100, 3)}')
    print(scores)

  def predict(self, test_pdf_path, result_csv_path):
    features_dictionary = Specification.get_features_dictionary(test_pdf_path)
    filtered_X = Specification.get_X_from_features_dictionary(features_dictionary, self.feature_keys_to_use)
    Y = self.clf_rf.predict(filtered_X)
    fully_featured_X = Specification.get_X_from_features_dictionary(features_dictionary)
    Specification.export_to_csv_from_XY(result_csv_path, fully_featured_X, Y)

  def visualize(self):
    dot_data = tree.export_graphviz(self.clf_dt, out_file=None,
                      feature_names=self.feature_keys_to_use,
                      class_names=['none', 'ss', 'se'],
                      filled=True, rounded=True,
                      special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("assets/tree_result")

  @staticmethod
  def get_XY_from_folder(training_csv_folder, feature_keys_to_use):
    res_X = []
    res_Y = []
    for file_name in os.listdir(training_csv_folder):
      if file_name.endswith(".csv"):
        X, Y = Specification.get_XY_from_csv(f'{training_csv_folder}/{file_name}', feature_keys_to_use)
        res_X.extend(X)
        res_Y.extend(Y)
    return res_X, res_Y

  @staticmethod
  def export_to_csv(pdf_path, csv_path):
    features_dictionary = Specification.get_features_dictionary(pdf_path)
    fully_featured_X = Specification.get_X_from_features_dictionary(features_dictionary)
    Specification.export_to_csv_from_XY(csv_path, fully_featured_X)

  @staticmethod
  def export_to_csv_from_XY(csv_path, X, Y = None):
    with open(csv_path, 'w', newline='') as csvfile:
      fieldnames = ['Y']
      fieldnames.extend(Specification.feature_keys)
      w = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
      w.writeheader()
      for i, _ in enumerate(X):
        w.writerow({
          'Y': (Specification.default_Y_value if Y is None else Y[i]),
          'page': X[i][0],
          'rel_x': X[i][1],
          'rel_y': X[i][2],
          'text': X[i][3],
          'len': X[i][4],
          'section_idx': X[i][5],
          'end_idx': X[i][6],
          'has_6_digits': X[i][7],
          'x': X[i][8],
          'y': X[i][9],
          })

  @staticmethod
  def get_features_dictionary(pdf_path):
    features_dictionary = {}
    reader = PdfReader(pdf_path)
    current_page_number = -1
    current_page = None
    def visitor_body(text, cm, tm, fontDict, fontSize):
      x = tm[4]
      y = tm[5]
      box = current_page.cropbox
      rel_x = int((x-float(box.left))/(float(box.right)-float(box.left))*100)
      rel_y = int((float(box.top)-y)/(float(box.top)-float(box.bottom))*100)
      for t in text.split("\n"):
        new_line_counter = Specification.get_new_line_counter(features_dictionary, current_page_number, rel_y)
        key = Specification.get_features_dictionary_key(current_page_number, rel_y, new_line_counter)
        if key in features_dictionary:
          features_dictionary[key]['text'] += t
        else:
          features_dictionary[key] = {
            'page': current_page_number,
            'rel_x': rel_x,
            'rel_y': rel_y,
            'text': t,
            'x': x,
            'y': y,
          }
          Specification.increment_new_line_counter(features_dictionary, current_page_number, rel_y)
    for i, page in enumerate(reader.pages):
      current_page = page
      current_page_number = i + 1
      page.extract_text(visitor_text=visitor_body)
    return Specification.post_process_features_dictionary(features_dictionary)

  @staticmethod
  def get_features_dictionary_key(page_number, y, extra):
    return f'{page_number}-{int(y)}-{extra}'

  @staticmethod
  def get_new_line_counter(features_dictionary, page_number, y):
    key = Specification.get_features_dictionary_key(page_number, y, 0)
    return features_dictionary[key]['new_line_count'] if key in features_dictionary else 0

  @staticmethod
  def increment_new_line_counter(features_dictionary, page_number, y):
    key = Specification.get_features_dictionary_key(page_number, y, 0)
    if 'new_line_count' in features_dictionary[key]:
      features_dictionary[key]['new_line_count'] += 1
    else:
      features_dictionary[key]['new_line_count'] = 1

  @staticmethod
  def post_process_features_dictionary(features_dictionary):
    keysToBeDeleted = []
    for key in features_dictionary:
      features = features_dictionary[key]
      text = features['text'].strip()
      if text == "":
        keysToBeDeleted.append(key)
      features['text'] = text
      features['len'] = len(text)
      features['section_idx'] = Specification.get_index(text, 'section')
      features['end_idx'] = Specification.get_index(text, 'end')
      features['has_6_digits'] = 1 if Specification.has_6_digits(text) else -1
      features_dictionary[key] = features
    for key in keysToBeDeleted:
      del features_dictionary[key]
    return features_dictionary

  @staticmethod
  def get_XY_from_csv(csv_path, feature_keys_to_use):
    X = []
    Y = []
    with open(csv_path, newline='') as csvfile:
      r = csv.DictReader(csvfile, delimiter='\t')
      for row in r:
        Y.append(row['Y'])
        features = []
        for feature_key in feature_keys_to_use:
          features.append(row[feature_key])
        X.append(features)
    return X, Y

  @staticmethod
  def get_X_from_features_dictionary(features_dictionary, feature_keys_to_use = None):
    X = []
    for key in features_dictionary:
      features = []
      if feature_keys_to_use is None:
        for feature_key in Specification.feature_keys:
          features.append(features_dictionary[key][feature_key])
      else:
        for feature_key in feature_keys_to_use:
          features.append(features_dictionary[key][feature_key])
      X.append(features)
    return X

  @staticmethod
  def get_index(str, substr):
    try:
      return str.lower().index(substr)
    except ValueError:
      return -1

  @staticmethod
  def has_6_digits(str):
    count = 0
    for ch in str:
      if ch.isdigit():
        count += 1
    return count == 6

feature_keys_to_use = ['len', 'section_idx', 'end_idx', 'rel_x', 'rel_y', 'has_6_digits']
training_csv_path = "assets/training_data.csv"
training_csv_folder = "/Users/jeehwancho/repos/pdf_extractor/assets/training_csvs"
test_pdf_path = "/Users/jeehwancho/repos/pdf_extractor/assets/test_pdfs/JW_MARRIOTT_SPECIFICATIONS.pdf"
result_csv_path = "assets/result.csv"

# Specification.export_to_csv(training_pdf_path, training_csv_path)
spec = Specification(feature_keys_to_use) # train based on csv
spec.score(training_csv_folder)
spec.train(training_csv_folder)
spec.predict(test_pdf_path, result_csv_path) # generate csv
spec.visualize()
print("done.")
