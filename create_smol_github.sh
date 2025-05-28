#!/bin/bash
git init
git add .
git commit -m "Initial commit"
gh repo create smol_llms_trainer --public --source=. --remote=origin --push

