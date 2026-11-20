# Repository publishing and attribution

This repository contains an independently written 3D, 7-DOF simulation inspired by https://github.com/Aashika-jaiswal/Robotic_arm_path_planning .

It is not a GitHub fork, and does not include the original author's code or image assets. If reusing upstream material, obtain the permission or license required by its author.

## Run

```powershell
git clone https://github.com/Jayashtandekar/robotic-arm-3d-7dof.git
cd robotic-arm-3d-7dof
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_7dof.py --gif
python -m unittest discover -s tests -v
```

Use `--trials 30` and multiple obstacle scenes for stronger statistical claims. The included three-trial benchmark is illustrative.
