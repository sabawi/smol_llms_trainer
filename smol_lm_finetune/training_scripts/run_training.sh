# training_scripts/run_training.sh
#!/bin/bash

# Activate virtual environment if not already active
# source ../smol_lm_env/bin/activate # Uncomment if you run this script directly without activating env first

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/run_finetuning.py"
LOG_FILE="$SCRIPT_DIR/../training_output/training_run.log"

ACTION=$1
RESUME_PATH_ARG=""

if [ "$ACTION" == "start" ]; then
    echo "Starting new training..."
    # Optional: Clear previous logs or outputs if desired, BE CAREFUL
    # rm -rf $SCRIPT_DIR/../training_output/checkpoints/*
    # rm -rf $SCRIPT_DIR/../training_output/logs/*
    # rm -rf $SCRIPT_DIR/../training_output/final_model/*
    # rm -rf $SCRIPT_DIR/../training_output/gguf_model/*
    nohup python -u "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
    echo "Training started in background. PID: $!. Log: $LOG_FILE"
    echo $! > "$SCRIPT_DIR/../training_output/trainer.pid"

elif [ "$ACTION" == "resume" ]; then
    if [ -f "$SCRIPT_DIR/../training_output/trainer.pid" ]; then
        echo "A training process might still be running (PID: $(cat "$SCRIPT_DIR/../training_output/trainer.pid")). Please pause/stop it first or ensure it has finished."
        # exit 1 # Or allow to proceed with resume if PID is stale
    fi
    RESUME_OPTION=$2
    if [ -z "$RESUME_OPTION" ] || [ "$RESUME_OPTION" == "auto" ]; then
        echo "Resuming training from latest checkpoint (auto)..."
        RESUME_PATH_ARG="--resume auto"
    elif [ -d "$RESUME_OPTION" ]; then # Check if it's a directory path
        echo "Resuming training from checkpoint: $RESUME_OPTION..."
        RESUME_PATH_ARG="--resume $RESUME_OPTION"
    else
        echo "Invalid resume option. Use 'auto' or provide a valid checkpoint directory."
        exit 1
    fi
    nohup python -u "$PYTHON_SCRIPT" $RESUME_PATH_ARG > "$LOG_FILE" 2>&1 &
    echo "Training resumed in background. PID: $!. Log: $LOG_FILE"
    echo $! > "$SCRIPT_DIR/../training_output/trainer.pid"

elif [ "$ACTION" == "pause" ]; then # Graceful pause is hard externally without script cooperation
    echo "Pausing training (sending SIGINT to allow graceful shutdown at checkpoint)..."
    if [ -f "$SCRIPT_DIR/../training_output/trainer.pid" ]; then
        PID=$(cat "$SCRIPT_DIR/../training_output/trainer.pid")
        if ps -p $PID > /dev/null; then
            kill -SIGINT $PID # Send Interrupt signal
            echo "SIGINT sent to PID $PID. The trainer should save a checkpoint and exit gracefully."
            echo "Wait for the process to terminate, then you can resume."
            # Optionally wait and check
            # timeout 300 tail --pid=$PID -f /dev/null # Wait up to 5 mins
            # if ps -p $PID > /dev/null; then
            #     echo "Process $PID still running. Consider a SIGTERM (kill $PID) or SIGKILL (kill -9 $PID) if it's stuck."
            # else
            #     rm "$SCRIPT_DIR/../training_output/trainer.pid"
            # fi
        else
            echo "PID $PID not found. Maybe training already stopped?"
            rm "$SCRIPT_DIR/../training_output/trainer.pid"
        fi
    else
        echo "Trainer PID file not found. Cannot pause."
    fi

elif [ "$ACTION" == "stop" ]; then # More forceful stop
    echo "Stopping training (sending SIGTERM)..."
     if [ -f "$SCRIPT_DIR/../training_output/trainer.pid" ]; then
        PID=$(cat "$SCRIPT_DIR/../training_output/trainer.pid")
        if ps -p $PID > /dev/null; then
            kill $PID # Send Terminate signal
            echo "SIGTERM sent to PID $PID."
            # rm "$SCRIPT_DIR/../training_output/trainer.pid" # Remove immediately or after confirmation
        else
            echo "PID $PID not found."
            rm "$SCRIPT_DIR/../training_output/trainer.pid"
        fi
    else
        echo "Trainer PID file not found."
    fi
elif [ "$ACTION" == "status" ]; then
    if [ -f "$SCRIPT_DIR/../training_output/trainer.pid" ]; then
        PID=$(cat "$SCRIPT_DIR/../training_output/trainer.pid")
        if ps -p $PID > /dev/null; then
            echo "Training is running with PID $PID."
            echo "Tail log: $LOG_FILE"
            tail -n 20 "$LOG_FILE"
        else
            echo "Training PID file exists, but process $PID is not running. It might have finished or crashed."
            echo "Last lines of log: $LOG_FILE"
            tail -n 20 "$LOG_FILE"
        fi
    else
        echo "Training does not appear to be running (no PID file)."
        if [ -f "$LOG_FILE" ]; then
             echo "Last lines of log: $LOG_FILE"
             tail -n 20 "$LOG_FILE"
        fi
    fi
else
    echo "Usage: $0 {start|resume [auto|checkpoint_path]|pause|stop|status}"
    exit 1
fi