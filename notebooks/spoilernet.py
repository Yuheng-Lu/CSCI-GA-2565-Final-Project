import torch


class WordAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(WordAttention, self).__init__()
        self.attention_transform = torch.nn.Linear(hidden_dim, hidden_dim)
        self.tanh_activation = torch.nn.Tanh()
        self.context_vector = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_weights = torch.nn.Softmax(dim=1)

    def forward(self, gru_outputs):
        transformed_attention = self.tanh_activation(
            self.attention_transform(gru_outputs)
        )

        attention_score = self.context_vector(transformed_attention)

        attention_distribution = self.attention_weights(attention_score)

        attended_output = torch.sum(attention_distribution * gru_outputs, dim=1)

        return attended_output


class SpoilerNet(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pretrained_embeddings):
        super(SpoilerNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = torch.nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=False
        )
        self.word_gru = torch.nn.GRU(
            embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True
        )

        self.word_attention = WordAttention(self.hidden_dim)

        self.sentence_gru = torch.nn.GRU(
            self.hidden_dim, self.hidden_dim, bidirectional=True, batch_first=True
        )
        self.dropout_layer = torch.nn.Dropout(0.5)  # Dropout as per the original paper
        self.final_linear = torch.nn.Linear(self.hidden_dim, 2)
        self.sigmoid_output = torch.nn.Sigmoid()

    def forward(self, inputs):
        embedded_text = self.embedding(inputs)

        word_encoding_output, _ = self.word_gru(embedded_text)

        combined_gru_outputs = (
            word_encoding_output[:, :, : self.hidden_dim]
            + word_encoding_output[:, :, self.hidden_dim :]
        )

        sentence_vector = self.word_attention(combined_gru_outputs).unsqueeze(1)

        sentence_encoding_output, _ = self.sentence_gru(sentence_vector)
        combined_sentence_output = (
            sentence_encoding_output[:, :, : self.hidden_dim]
            + sentence_encoding_output[:, :, self.hidden_dim :]
        )

        output_after_dropout = self.dropout_layer(combined_sentence_output)
        class_scores = self.final_linear(output_after_dropout)  # Raw class scores
        return class_scores
