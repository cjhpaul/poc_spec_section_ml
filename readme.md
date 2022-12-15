## Machine Learning & Specification Section Extraction

### Goal
- Evaluate feasibility the spec section extraction via POC
- Identify limits
- Identify ways of integrating into existing products

### Notes
- As a python newbie, this code is not following the python best practices.
- It's written in python because of the popular ML libraries like sklearn
- I didn't spend much on comparing different classifiers. I started with decision tree as it can visuallize the decision tree. Then random forest because... it's a forest rather than a tree :D
- It transforms pdf into csv, because (1) I want to inspect data that's persisted somewhere, (2) there are many csv editors out there (no need to implement FE/UI for POC).
- Features: x & y coordinates of a text, indices of some key words (e.g., section), and length of a text.
- Accuracy is >99% via cross-validation with standard configuration (5 splits and 25% test). But the accuracy might not be accurate (to be discussed in limits section)
- Labels: none, ss (section starts), se (section ends)
- Test files are not checked into this repo due to confidentiality.

### Limits/difficulties
- The pdf reader used in POC is far from perfect. Time to time it acts out on line breakers (e.g., SOME_TEXT_ABOVE\nSECTION\n123456 instead of SECTION 123456). The x & y coordinates of a text sometime jumps around. Definitely need to evaludate other pdf readers.
- Manually labelling test data is boring and painful
- 99% accuracy could be misleading because (1) only 2 pdf files were used (total number of pages is ~2k), (2) distribution of classifications/labels are not well balanced (e.g. the number of start of section is a lot smaller than texts not classified for anything). In that sense we can use a [stratification](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-with-stratification-based-on-class-labels) as cross-validation strategy.

### Possible applications (given that accuracy is stable)
- Deploy a service that periodically pulls user-confirmed spec sections, and train on them. Keep it in the dark until numbers look good. Eventually it generate recommendations to users if the existing service fails to extract data from specification.

### Tech debts
- Explore different pdf readers
- Write a method to update existing csv files when features get updated
- Refactoring (e.g., properly use class variables/methods)
- Explore more features and functionalities to support other classifications as necessary
- Look for options that are python equivalent to npm for javascript or bundler for ruby.
