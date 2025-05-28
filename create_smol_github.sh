# Step 1: Go to your local project directory
cd ~/Development/smol_llm

# Step 2: Initialize a new Git repository
git init

# Step 3: Add all existing files to the repo
git add .

# Step 4: Commit the files
git commit -m "Initial commit"

# Step 5: Create the GitHub repo via SSH
# (You must have GitHub CLI `gh` installed and authenticated)

gh repo create smol_llms_trainer --public --source=. --remote=origin --ssh --push

