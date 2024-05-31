{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "## `1.` Importing Libraries"
      ],
      "metadata": {
        "id": "KqG2dN3H4YC0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pip install LibRecommender"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hUPRA3Qt4PKk",
        "outputId": "aaa4bf4c-86f3-4b19-c77c-3d4094410600"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: LibRecommender in /usr/local/lib/python3.10/dist-packages (1.4.0)\n",
            "Requirement already satisfied: gensim>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from LibRecommender) (4.3.2)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from LibRecommender) (4.66.4)\n",
            "Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.10/dist-packages (from gensim>=4.0.0->LibRecommender) (1.25.2)\n",
            "Requirement already satisfied: scipy>=1.7.0 in /usr/local/lib/python3.10/dist-packages (from gensim>=4.0.0->LibRecommender) (1.11.4)\n",
            "Requirement already satisfied: smart-open>=1.8.1 in /usr/local/lib/python3.10/dist-packages (from gensim>=4.0.0->LibRecommender) (6.4.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from libreco.data import DatasetFeat, split_by_ratio_chrono\n",
        "from libreco.algorithms import NCF\n",
        "from libreco.evaluation import evaluate\n",
        "import tensorflow as tf\n",
        "import pickle"
      ],
      "metadata": {
        "id": "dbOpAM3-4J3X"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## `2.` Loading Dataset"
      ],
      "metadata": {
        "id": "GDLmK6tJ5K3_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('/content/drive/MyDrive/movies_rs (1).csv')"
      ],
      "metadata": {
        "id": "gsfGZtry_o0U"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##`3.` Split dataset\n"
      ],
      "metadata": {
        "id": "qnhmzZjE64rd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data, validation_data = split_by_ratio_chrono(df, test_size=0.2)\n",
        "val_data, test_data = split_by_ratio_chrono(validation_data, test_size=0.5)"
      ],
      "metadata": {
        "id": "E43JRbps6yZY"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_length = len(train_data)\n",
        "validation_length = len(val_data)\n",
        "test_length = len(test_data)\n",
        "\n",
        "# Print the lengths\n",
        "print(f\"Length of the train set: {train_length}\")\n",
        "print(f\"Length of the validation set: {validation_length}\")\n",
        "print(f\"Length of the test set: {test_length}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N_8AjiyzwvZK",
        "outputId": "0582284d-a604-41e0-bbdf-bf0cc7c5f213"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Length of the train set: 82140\n",
            "Length of the validation set: 9998\n",
            "Length of the test set: 8632\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## `4.` Build train and test sets"
      ],
      "metadata": {
        "id": "GfzsRmyZ7L5I"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data, data_info = DatasetFeat.build_trainset(train_data)\n",
        "val_data = DatasetFeat.build_testset(val_data)\n",
        "test_data = DatasetFeat.build_testset(test_data)"
      ],
      "metadata": {
        "id": "k_hfUG937Ua2"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## `5.` Build NCF Model"
      ],
      "metadata": {
        "id": "3hx-jC6w7c9T"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "hidden_units = [int(unit) for unit in \"128,64,32\".split(',')]\n",
        "\n",
        "ncf = NCF(\n",
        "    task=\"rating\",\n",
        "    data_info=data_info,\n",
        "    embed_size=16,\n",
        "    n_epochs=10,\n",
        "    lr=0.001,\n",
        "    batch_size=256,\n",
        "    num_neg=4,\n",
        "    use_bn=True,\n",
        "    dropout_rate=None,\n",
        "    hidden_units=hidden_units,\n",
        "    reg=None,\n",
        ")"
      ],
      "metadata": {
        "id": "dLUpsbfQ7g80"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##`6.`Train NCF Model"
      ],
      "metadata": {
        "id": "LVBagvoy7msZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "ncf.fit(train_data, neg_sampling=False, verbose=2, eval_data=val_data, metrics=[\"rmse\", \"mae\"])"
      ],
      "metadata": {
        "id": "TEsvkHXH7kAN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "503d2664-2bed-4f38-e29c-d71894b1f2d8"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training start time: \u001b[35m2024-05-21 20:18:15\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/libreco/layers/dense.py:31: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).\n",
            "  net = tf.layers.batch_normalization(net, training=is_training)\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.10/dist-packages/keras/src/layers/normalization/batch_normalization.py:883: _colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Colocations handled automatically by placer.\n",
            "/usr/local/lib/python3.10/dist-packages/libreco/layers/dense.py:39: UserWarning: `tf.layers.batch_normalization` is deprecated and will be removed in a future version. Please use `tf.keras.layers.BatchNormalization` instead. In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` should not be used (consult the `tf.keras.layers.BatchNormalization` documentation).\n",
            "  net = tf.layers.batch_normalization(net, training=is_training)\n",
            "train: 100%|██████████| 1611/1611 [00:06<00:00, 261.89it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1 elapsed: 6.165s\n",
            "\t \u001b[32mtrain_loss: 0.0099\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 16.52it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0005\n",
            "\t eval mae: 0.0003\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:04<00:00, 352.53it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 2 elapsed: 4.583s\n",
            "\t \u001b[32mtrain_loss: 0.0\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 72.28it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0009\n",
            "\t eval mae: 0.0005\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:03<00:00, 403.85it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 3 elapsed: 3.995s\n",
            "\t \u001b[32mtrain_loss: 0.0001\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 91.95it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0097\n",
            "\t eval mae: 0.0053\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:05<00:00, 308.38it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 4 elapsed: 5.236s\n",
            "\t \u001b[32mtrain_loss: 0.0001\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 67.50it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0014\n",
            "\t eval mae: 0.0008\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:05<00:00, 314.98it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 5 elapsed: 5.123s\n",
            "\t \u001b[32mtrain_loss: 0.0001\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 101.79it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0008\n",
            "\t eval mae: 0.0004\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:04<00:00, 366.54it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 6 elapsed: 4.405s\n",
            "\t \u001b[32mtrain_loss: 0.0\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 104.43it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0038\n",
            "\t eval mae: 0.0023\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:05<00:00, 309.07it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 7 elapsed: 5.223s\n",
            "\t \u001b[32mtrain_loss: 0.0001\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 75.32it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0005\n",
            "\t eval mae: 0.0003\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:05<00:00, 310.94it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 8 elapsed: 5.188s\n",
            "\t \u001b[32mtrain_loss: 0.0\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 99.04it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0002\n",
            "\t eval mae: 0.0001\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:03<00:00, 412.27it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 9 elapsed: 3.916s\n",
            "\t \u001b[32mtrain_loss: 0.0\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 72.60it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0001\n",
            "\t eval mae: 0.0001\n",
            "==============================\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "train: 100%|██████████| 1611/1611 [00:04<00:00, 364.91it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 10 elapsed: 4.424s\n",
            "\t \u001b[32mtrain_loss: 0.0\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 59.18it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\t eval rmse: 0.0002\n",
            "\t eval mae: 0.0001\n",
            "==============================\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## `7.` Evaluate NCF Model"
      ],
      "metadata": {
        "id": "pUS_s7jj8Bgr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Evaluation on test data:\")\n",
        "evaluate(ncf, test_data, neg_sampling=False, eval_batch_size=8192, metrics=[\"rmse\", \"mae\"])"
      ],
      "metadata": {
        "id": "VNEkZOfQ7s8q",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ea3e85d4-04eb-45f3-defc-9e882b3b4ab8"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Evaluation on test data:\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "eval_pointwise: 100%|██████████| 2/2 [00:00<00:00, 80.67it/s]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'rmse': 0.00019941166, 'mae': 0.0001079182}"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##`8.` Get top 10 movie recommendations for a user"
      ],
      "metadata": {
        "id": "0RBIpv9f80Ar"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "user_id = 5  # Example user ID\n",
        "top_k = 10\n",
        "rec_movies = ncf.recommend_user(user=user_id, n_rec=top_k)\n",
        "rec_movies[user_id]"
      ],
      "metadata": {
        "id": "Purugfsw871D",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1003a3ca-2bcc-4e21-ebd8-7939346edc06"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([  852, 89774,   490,  6104, 89492, 52973, 59900,  3421, 36525,\n",
              "       47810])"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df[df['item'] == 1608]['title'].values[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "1rpEYuHH2eeu",
        "outputId": "f05acea0-a5d0-45dd-e499-223f9b2f4a2b"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Air Force One'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def get_movie_title(movie_id, movies_df):\n",
        "    title = df[df['item'] == movie_id]['title'].values[0]\n",
        "    return title\n",
        "\n",
        "# Print the movie titles\n",
        "print(\"Top 10 movie recommendations for user ID {}:\".format(user_id))\n",
        "for movie_id in rec_movies[user_id]:\n",
        "    title = get_movie_title(movie_id, df)\n",
        "    print(f\"Movie ID: {movie_id}, Title: {title}\")"
      ],
      "metadata": {
        "id": "3wh8WsQF9AgE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5e116122-0102-4646-d974-a8650042ed74"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Top 10 movie recommendations for user ID 5:\n",
            "Movie ID: 852, Title: Tin Cup\n",
            "Movie ID: 89774, Title: Warrior\n",
            "Movie ID: 490, Title: Malice\n",
            "Movie ID: 6104, Title: Monty Python Live at the Hollywood Bowl\n",
            "Movie ID: 89492, Title: Moneyball\n",
            "Movie ID: 52973, Title: Knocked Up\n",
            "Movie ID: 59900, Title: You Don't Mess with the Zohan\n",
            "Movie ID: 3421, Title: Animal House\n",
            "Movie ID: 36525, Title: Just Like Heaven\n",
            "Movie ID: 47810, Title: Wicker Man, The\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##`8.` Save NCF Model"
      ],
      "metadata": {
        "id": "gZeCPJmP3zNk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#saving model\n",
        "ncf.save(\"ncf_user_model.h5\", model_name=\"ncf_model\")\n",
        "\n",
        "# Save the data_info object to a file\n",
        "with open('data_info.pkl', 'wb') as f:\n",
        "    pickle.dump(data_info, f)"
      ],
      "metadata": {
        "id": "2GuSnxYCCysO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f6ea786c-bcaa-4a42-8cce-43232fffbefe"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "file folder ncf_user_model.h5 doesn't exists, creating a new one...\n"
          ]
        }
      ]
    }
  ]
}