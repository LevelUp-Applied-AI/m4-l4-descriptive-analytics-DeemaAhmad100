# Findings Report - Descriptive Analytics  
**Hashemite Technical University Student Performance**

## 1. Dataset Description
- **Shape**: 2000 rows × 10 columns (2000 student records)
- **Key Columns**: department, semester, course_load, study_hours_weekly, gpa, attendance_pct, has_internship, commute_minutes, scholarship
- **Data Quality Issues**:
  - `commute_minutes`: 181 missing values (9.05%) → imputed with median (25 minutes)
  - `scholarship`: 389 missing values (19.45%) → filled with "None"
  - No rows were dropped during cleaning. Final dataset remains 2000 rows.

## 2. Key Distribution Findings
- GPA is left-skewed with most students clustering between 2.5–3.5.
- Study hours per week show reasonable variation.
- Clear differences in GPA distributions across departments (see `gpa_by_department.png`).
- Most students receive "None" or Merit-based scholarships.

## 3. Notable Correlations
- **Strongest correlation**: Between `study_hours_weekly` and `gpa` (**r = 0.639**).  
  Students who study more hours per week tend to achieve higher GPAs.
- Correlation between `attendance_pct` and `gpa` is very weak (**r = 0.041**).  
  This suggests that class attendance alone is not a strong predictor of academic success in this dataset (see `attendance_vs_gpa.png`).
- Correlation is not causation — other factors may influence these relationships.

## 4. Hypothesis Test Results

**Hypothesis 1**: Students with internships have a higher GPA than students without internships.  
- **Test Used**: Independent samples t-test  
- **t-statistic**: 14.2288  
- **p-value**: 0.000000  
- **Cohen’s d**: 0.690 (medium to large effect size)  
- **Interpretation**: Statistically significant (p < 0.001). Students with internships have significantly higher GPAs.

**Hypothesis 2**: Scholarship status is associated with department.  
- **Test Used**: Chi-square test of independence  
- **Chi-square statistic**: 17.1358  
- **p-value**: 0.376862  
- **Degrees of freedom**: 16  
- **Interpretation**: Not statistically significant (p > 0.05). There is no significant association between scholarship type and department.

## 5. Actionable Recommendations for the University

1. **Promote Internship Opportunities**  
   Since students with internships show significantly higher GPAs (large effect size), the university should expand internship programs and encourage more students to participate.

2. **Focus on Self-Study Skills**  
   The strong correlation between study hours and GPA, combined with the weak correlation with attendance, suggests the university should invest in workshops on effective self-study techniques and time management.

3. **Review Scholarship Allocation**  
   Since scholarship type is not significantly associated with specific departments, the university could consider making scholarship criteria more merit-based or need-based across all departments rather than department-specific.

---

**References to Visuals**:
- GPA Distribution: `output/gpa_distribution.png`
- GPA by Department: `output/gpa_by_department.png`
- Correlation Heatmap: `output/correlation_heatmap.png`
- Study Hours vs GPA: `output/study_hours_vs_gpa.png`
- Attendance vs GPA: `output/attendance_vs_gpa.png`