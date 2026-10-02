FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY handler.py ${LAMBDA_TASK_ROOT}/
COPY localpulse ${LAMBDA_TASK_ROOT}/localpulse
COPY static ${LAMBDA_TASK_ROOT}/static

CMD ["handler.lambda_handler"]
