        
FROM python:3.7-stretch

RUN pip3 install numpy==1.20.1 pandas scikit-learn scipy==1.4.0 git+https://github.com/slundberg/shap.git
    
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
