import pandas as pd
import matplotlib.pyplot as plt
base_gru_acc = pd.read_csv('outputs/base_gru/acc_recorder.csv')
base_lstm_acc = pd.read_csv('outputs/base_lstm/acc_recorder.csv')
lstm_acc = pd.read_csv('outputs/lstm/acc_recorder.csv')
lstm_att_acc = pd.read_csv('outputs/lstm_att/acc_recorder.csv')

base_gru_loss = pd.read_csv('outputs/base_gru/loss_recorder.csv')
base_lstm_loss = pd.read_csv('outputs/base_lstm/loss_recorder.csv')
lstm_loss = pd.read_csv('outputs/lstm/loss_recorder.csv')
lstm_att_loss = pd.read_csv('outputs/lstm_att/loss_recorder.csv')

# Plotting Test Loss
plt.figure(figsize=(10, 5))
plt.plot(base_gru_loss['test loss'], label='Base LSTM', color='blue')
plt.plot(base_lstm_loss['test loss'], label='Base GRU', color='green')
plt.plot(lstm_loss['test loss'], label='LSTM_teacher', color='red')
plt.plot(lstm_att_loss['test loss'], label='LSTM_teacher_att', color='orange')
plt.title('Test Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('test_loss_plot.png')  # Save the plot to a file
plt.show()

# Plotting Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(base_gru_acc['val acc'], label='Base LSTM', color='blue')
plt.plot(base_lstm_acc['val acc'], label='Base GRU', color='green')
plt.plot(lstm_acc['val acc'], label='LSTM_teacher', color='red')
plt.plot(lstm_att_acc['val acc'], label='LSTM_teacher_att', color='orange')
plt.title('Validation Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('val_acc_plot.png')  # Save the plot to a file
plt.show()

lstm_acc = pd.read_csv('outputs/lstm_large/acc_recorder.csv')
lstm_loss = pd.read_csv('outputs/lstm_large/loss_recorder.csv')

plt.figure(figsize=(10, 5))
plt.plot(lstm_acc['train acc'], label='training', color='blue')
plt.plot(lstm_acc['val acc'], label='validation', color='red')
plt.title(' Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lstm_acc.png')  # Save the plot to a file
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(lstm_loss['train loss'], label='training', color='blue')
plt.plot(lstm_loss['test loss'], label='validation', color='red')
plt.title(' lost vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('lost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lstm_lost.png')  # Save the plot to a file
plt.show()

