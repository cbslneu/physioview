# TODO

A to-do list of planned features, improvements, and tasks for HeartView.

## ✨ Features to Add
- [x] **DB:** Batch post-processing functionality with a UI component [@nmy2103]
- [x] **DB:** Automated beat correction functionality with UI component
- [ ] **BE:** Functionality to flexibly edit added/deleted beats and unusable segments [@tchen94]
- [x] **DB:** EDA processing and quality assessment functionality [@nmy2103]

## 🐛 Bugs to Fix
- [ ] **DB:** Recreate Beat Editor files after changing the segment size 
  when preprocessing the same uploaded file [@nmy2103]
- [ ] **DB:** Fix rendering mismatch of detected beats on filtered vs. raw 
  signals


## 🛠️ Improvements
- [x] **BE:** Migrate codebase to TypeScript
- [ ] **BE:** Document or refactor to avoid confusion in the use of undo button


## 📕 Documentation
- [ ] Write usage examples in `examples.rst` for:
  - [ ] Data pre-processing
  - [ ] Signal quality assessment
  - [ ] Data post-processing