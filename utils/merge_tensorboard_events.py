import sys
import tensorflow as tf

# Check usage
if len(sys.argv) < 4:
    print("Usage: script_name.py events_file1 events_file2 combined_events_file")
    sys.exit()

# Paths to the TensorBoard events files and the output combined events file
events_file1, events_file2, combined_events_file = sys.argv[1], sys.argv[2], sys.argv[3]

highest_step = 0
finished_first_file = False

# Function to read events from a file and write them to another file
def read_and_write_events(input_file, output_writer):
    global highest_step, finished_first_file
    try:
        event_iterator = tf.compat.v1.train.summary_iterator(input_file)
        for event in event_iterator:
            if not finished_first_file:
                if event.step > highest_step:
                    highest_step = event.step
                if event.step < highest_step:
                    finished_first_file = True
            else:
                event.step += highest_step + 1

            output_writer.write(event.SerializeToString())
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        sys.exit(1)

# Write the combined events to the output file
try:
    with tf.io.TFRecordWriter(combined_events_file) as writer:
        read_and_write_events(events_file1, writer)
        read_and_write_events(events_file2, writer)
except Exception as e:
    print(f"Error writing to {combined_events_file}: {e}")
    sys.exit(1)

print(f"Combined events saved to {combined_events_file}")
