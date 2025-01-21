#!/bin/bash

# List available restore points
list_restore_points() {
    echo "Available restore points:"
    ls -l backups/restore_point_* | awk '{print $9}'
}

# Restore from a specific point
restore_from_point() {
    local restore_point=$1
    if [ ! -d "$restore_point" ]; then
        echo "Error: Restore point not found: $restore_point"
        exit 1
    fi
    
    echo "Restoring from: $restore_point"
    cp -r "$restore_point"/* .
    echo "Restore completed successfully!"
}

# Main script
if [ "$1" == "list" ]; then
    list_restore_points
elif [ -n "$1" ]; then
    restore_from_point "$1"
else
    echo "Usage:"
    echo "  ./restore.sh list              # List available restore points"
    echo "  ./restore.sh <restore_point>   # Restore from specified point"
fi 