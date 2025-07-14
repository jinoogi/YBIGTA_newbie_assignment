
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
if ! command -v conda &> /dev/null; then
    echo "[INFO] miniconda가 설치되어 있지 않습니다. 설치를 시작합니다."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    source $HOME/miniconda/bin/activate
else
    echo "[INFO] conda가 이미 설치되어 있습니다."
fi

# Conda 환셩 생성 및 활성화
## TODO
conda create --name myenv python==3.11 -y
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy > /dev/null 2>&1

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    fname="${file%%.py}"
    input_file="../input/${fname}_input"
    output_file="../output/${fname}_output"
    if [ -f "$input_file" ]; then
        python "$file" < "$input_file" > "$output_file"
    else
        echo "[INFO] 입력 파일이 없습니다: $input_file"
    fi
done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
cd ..
mypy submission/*.py > mypy_log.txt

# conda.yml 파일 생성
## TODO
conda env export --name myenv > conda.yml

# 가상환경 비활성화
## TODO
conda deactivate