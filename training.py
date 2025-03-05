import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os

from help_func import self_feeding, enc_self_feeding
from loss_func import total_loss
from nn_structure import AUTOENCODER

def get_model_path(i):
    path1 = f"/home/trarity/master/koopman_control/data/Autoencoder_model_params{i}.pth"
    path2 = f"C:/Users/jokin/Desktop/Uni/Aalborg/Master/Masters_Thesis/Path/Autoencoder_model_params{i}.pth"
    path3 = f"/content/drive/My Drive/Colab Notebooks/Autoencoder_model_params{i}.pth"
    path4 = f"/content/drive/MyDrive/Colab Notebooks/Autoencoder_model_params{i}.pth"
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    elif os.path.exists(path3):
        return path3
    else:
        return path4
def trainingfcn(eps, lr, batch_size, S_p, T, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder, train_tensor_unforced, train_tensor_forced, test_tensor, M):

  train_unforced_dataset = TensorDataset(train_tensor_unforced)
  train_unforced_loader = DataLoader(train_unforced_dataset, batch_size=batch_size, shuffle=True)

  train_forced_dataset = TensorDataset(train_tensor_forced)
  train_forced_loader = DataLoader(train_forced_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = TensorDataset(test_tensor)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  Model_path = []
  Models_loss_list = []
  Test_loss_list = []
  Running_Losses_Array = []
  c_m = 0

  Model_path = [get_model_path(i) for i in range(M)]

  for model_path_i in Model_path:

      # Instantiate the model and optimizer afresh
      model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder)
      loss_list = []
      running_loss_list = []
      nan_found = False  # Flag to detect NaNs

      print(f"Training unforced system dynamics, for Model: {c_m}")
      #First train the unforced system so do not compute
      set_requires_grad(list(model.u_Encoder_In.parameters()) +
                        list(model.u_Encoder_Hidden.parameters()) +
                        list(model.u_Encoder_out.parameters()) +
                        list(model.u_Koopman.parameters()) +
                        list(model.u_Decoder_In.parameters()) +
                        list(model.u_Decoder_Hidden.parameters()) +
                        list(model.u_Decoder_out.parameters()), requires_grad=False)

      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

      for e in range(eps):
          running_loss = 0.0
          for (batch_x,) in train_unforced_loader:

              optimizer.zero_grad()
              loss = total_loss_unforced(alpha, batch_x, Num_meas, Num_x_Obsv, T, S_p, model)

              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)


          avg_loss = running_loss / len(train_unforced_loader)
          loss_list.append(avg_loss)
          running_loss_list.append(running_loss)
          print(f'Epoch {e+1}, Model: {c_m}, Running loss: {running_loss:.3e}')
          current_lr = optimizer.param_groups[0]['lr']
          print(f'Current learning rate: {current_lr:.8f}')

          # Save the model parameters at the end of each epoch
          torch.save(model.state_dict(), model_path_i)


      print(f"Training input influence on dynamics, for Model: {c_m}")
      set_requires_grad(model.parameters(), requires_grad=False) # Set all parames to not train
      #Enable training of forced system
      set_requires_grad(list(model.u_Encoder_In.parameters()) +
                        list(model.u_Encoder_Hidden.parameters()) +
                        list(model.u_Encoder_out.parameters()) +
                        list(model.u_Koopman.parameters()) +
                        list(model.u_Decoder_In.parameters()) +
                        list(model.u_Decoder_Hidden.parameters()) +
                        list(model.u_Decoder_out.parameters()), requires_grad=True)


      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

      for e in range(eps):
          running_loss = 0.0
          for (batch_x,) in train_forced_loader:

              optimizer.zero_grad()
              loss = total_loss_forced(alpha, batch_x, Num_meas, Num_x_Obsv, T, S_p, model)

              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)


          avg_loss = running_loss / len(train_forced_loader)
          loss_list.append(avg_loss)
          running_loss_list.append(running_loss)
          print(f'Epoch {e+1}, Model: {c_m}, Running loss: {running_loss:.3e}')
          current_lr = optimizer.param_groups[0]['lr']
          print(f'Current learning rate: {current_lr:.8f}')

          # Save the model parameters at the end of each epoch
          torch.save(model.state_dict(), model_path_i)


      Models_loss_list.append(running_loss)
      Running_Losses_Array.append(running_loss_list)
      torch.save(model.state_dict(), model_path_i)

      for (batch_x,) in test_loader:
        [traj_prediction, loss] = enc_self_feeding(model, batch_x, Num_meas)
        running_loss += loss.item()

      avg_loss = running_loss / len(test_loader)
      print(f'Test Data w/Model {c_m + 1}, Avg Loss: {avg_loss:.10f}, Running loss: {running_loss:.3e}')
      Test_loss_list.append(running_loss)
      c_m += 1

  # Find the best of the models
  Lowest_loss = min(Models_loss_list)
  Lowest_test_loss = min(Test_loss_list)

  Lowest_loss_index = Models_loss_list.index(Lowest_loss)
  print(f"The best model has a running loss of {Lowest_loss} and is model nr. {Lowest_loss_index + 1}")

  Lowest_test_loss_index = Test_loss_list.index(Lowest_test_loss)
  print(f"The best model has a test running loss of {Lowest_test_loss} and is model nr. {Lowest_test_loss_index + 1}")

  Best_Model = Model_path[Lowest_test_loss_index]

  return Lowest_loss, Lowest_test_loss, Best_Model
