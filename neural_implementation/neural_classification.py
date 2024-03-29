import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


def g_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def g_sigmoid_derivative(z):
    return g_sigmoid(z) * (1 - g_sigmoid(z))


def hypothesis_neural(neural_model, input_data, vectorized=False):
    a_l = input_data

    for l in range(neural_model.shape[0]):
        z_lp1 = neural_model[l].dot(a_l)
        a_lp1 = g_sigmoid(z_lp1)

        # Ako nije poslednji sloj, dodaj bias jedinicu
        if l != neural_model.shape[0] - 1:
            a_lp1 = np.concatenate((np.array([1]), a_lp1))

        a_l = a_lp1

    if a_l.shape[0] == 1 and vectorized is False:
        return a_l[0]

    return a_l


def unroll_matrix(matrix):
    return matrix.ravel()


def unroll_matrix_array(matrix_vec):
    unrolled = np.array([])
    for matrix in matrix_vec:
        unrolled = np.concatenate((unrolled, unroll_matrix(matrix)), axis=None)

    return np.array(unrolled).ravel()


def roll_vec_to_matrix(vec, matrix_size):
    rows = matrix_size[0]
    cols = matrix_size[1]

    if vec.shape[0] != rows * cols:
        raise ValueError(f'Nekorektno razvijanje vektora velicine {vec.shape[0]} u matricu'
                         f' dimenzija {rows}x{cols} (matrica od {rows * cols} elemenata)')

    return vec.reshape(rows, cols)


def roll_vec_to_matrix_array(long_vec, matrix_sizes):
    matrix_array = []
    prev_rows = 0
    prev_cols = 0

    for matrix_size in matrix_sizes:
        rows = matrix_size[0]
        cols = matrix_size[1]

        i = prev_rows * prev_cols
        j = rows * cols
        matrix_array.append(roll_vec_to_matrix(long_vec[i: i + j], matrix_size))

        prev_rows = rows
        prev_cols = cols

    return np.array(matrix_array)


def get_matrix_sizes(matrix_array):
    return [matrix.shape for matrix in matrix_array]


def regularization_depr(neural_model, N_set_size, lambda_param=0):
    """ZASTARELA, neefikasna"""
    if lambda_param == 0:
        return 0

    reg = 0
    for l in range(neural_model.shape[0]):
        for i in range(neural_model[l].shape[0]):
            for j in range(1, neural_model[l].shape[1]):  # ne kaznjavamo bias jedinice
                reg += (neural_model[l][i][j]) ** 2

    return lambda_param * reg / (2 * N_set_size)


def regularization(neural_model, N_set_size, lambda_param=0):
    if lambda_param == 0:
        return 0

    reg = 0
    for l in range(neural_model.shape[0]):
        reg += (neural_model[l]**2).sum()
        reg -= (neural_model[l][:, 0]**2).sum()

    return lambda_param * reg / (2 * N_set_size)


def loss_instance(y_true, y_predict):
    out_size = y_true.shape[0]

    loss = 0
    for k in range(out_size):
        y = y_true[k]
        y_pred = y_predict[k]

        if y == 0:
            # Ignorisemo y*log(y_pred) clan, racunamo -log(1 - y_pred)
            loss -= np.log(1 - y_pred)
        elif y == 1:
            # Ignorisemo (1 - y)*log(1 - y_pred) clan, racunamo -log(y_pred)
            loss -= np.log(y_pred)
        else:
            raise ValueError(f'Incorrect class label {y}')

    return loss


def loss_logistic(X_data, y_data, neural_model, lambda_param=0):
    N = X_data.shape[0]
    out_size = y_data.shape[1]
    loss = 0

    if out_size == 2:
        raise ValueError('Nekorektni ulazni podaci: neocekivana duzina 2 ciljnog vektora')

    for i in range(N):
        yi = y_data[i]
        yi_pred = hypothesis_neural(neural_model, X_data[i], vectorized=True)

        loss += loss_instance(yi, yi_pred)

    return loss / N + regularization(neural_model, N, lambda_param)


def train_test_split(X_data, y_data, ratio='0.7 : 0.3'):
    set_sizes = ratio.split(':')
    set_sizes[0] = float(set_sizes[0].strip())
    set_sizes[1] = float(set_sizes[1].strip())

    N = X_data.shape[0]
    m = X_data.shape[1] - 1

    train_size = int(np.floor(N * set_sizes[0]))
    test_size = N - train_size
    print(f'Set sizes: \ntrain_size: {train_size}, test_size: {test_size}')

    merged_dataset = np.ones((N, m + 2))
    X_shuffled = np.zeros((N, m + 1))
    y_shuffled = np.zeros((N, 1))

    for i in range(N):
        merged_dataset[i] = np.append(X_data[i], y_data[i])

    np.random.shuffle(merged_dataset)

    for i in range(N):
        X_shuffled[i] = merged_dataset[i][:-1]
        y_shuffled[i] = merged_dataset[i][-1]

    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]

    X_test = X_shuffled[train_size:]
    y_test = y_shuffled[train_size:]

    return X_train, y_train, X_test, y_test


def plot_learning_curves(X_train, y_train, X_test, y_test, classifier):
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    losses_train = []
    losses_test = []

    # Napomena: loss_train i loss_test se racunaju BEZ regularizacionog izraza, tj. za lambda=0
    for N in range(1, N_train):
        loss_train_N = classifier.loss(X_train[:N], y_train[:N])  # <- lambda_param=0
        losses_train.append(loss_train_N)

        if N < N_test:
            loss_test_N = classifier.loss(X_test[:N], y_test[:N])  # <- lambda_param=0
            losses_test.append(loss_test_N)

    plt.plot(range(1, N_train), losses_train)
    plt.plot(range(1, N_test), losses_test, color='red')

    plt.xlabel('Training set size')
    plt.ylabel('Loss')

    plt.legend(['loss_train', 'loss_test'])
    plt.title('Learning curves')

    plt.show()


def gradient_checking(X_data, y_data, neural_model, lambda_param=0, eps=1e-4):
    w_unrolled = unroll_matrix_array(neural_model)
    n = len(w_unrolled)

    grad_approx = np.zeros((n,))
    matrix_sizes = get_matrix_sizes(neural_model)

    for i in range(n):
        # OBAVEZNO MORA KOPIJA VEKTORA, inace imamo dve reference na isti vektor!!!
        w_plus = w_unrolled.copy()
        w_minus = w_unrolled.copy()

        w_plus[i] += eps
        w_minus[i] -= eps

        w_plus_rolled = roll_vec_to_matrix_array(w_plus, matrix_sizes)
        w_minus_rolled = roll_vec_to_matrix_array(w_minus, matrix_sizes)

        loss_plus = loss_logistic(X_data, y_data, w_plus_rolled, lambda_param)
        loss_minus = loss_logistic(X_data, y_data, w_minus_rolled, lambda_param)

        grad_approx[i] = (loss_plus - loss_minus) / (2 * eps)

    return grad_approx


def gradient_descent(X_data,
                     y_data,
                     neural_network,
                     alpha=0.01,
                     num_iter=1000,
                     lambda_param=0,
                     model_init_bounds=(-1e-1, 1e-1),
                     plot=False,
                     gc_it=-1,
                     gc_log_path=None):

    loss_history = np.zeros((num_iter, 1))
    matrix_sizes = get_matrix_sizes(neural_network.model)

    # Pre nego sto zapocnemo trening modela potrebno je da se on inicijalizuje.
    # Za to se koristi random inicijalizacija
    neural_network.initialize_model(model_init_bounds=model_init_bounds)
    w = neural_network.unroll_model()

    f = None
    logged = False

    if gc_it > 0 and gc_log_path is not None:
        try:
            f = open(gc_log_path, 'w')
            logged = True
        except IOError:
            print('Log file not found: logging to STDOUT ...')

    for i in range(num_iter):
        loss, gradient = neural_network.backpropagation(X_data, y_data, lambda_param)
        loss_history[i] = loss

        if gc_it > 0 and (i == 0 or (i + 1) % gc_it == 0):
            grad_approx = gradient_checking(X_data, y_data, neural_network.model, lambda_param=lambda_param)

            diff = abs(gradient - grad_approx)
            out = f'Iteration {i + 1}:\n############################\nGradient = \n{gradient}\n\nGradient ' \
                  f'approximate =\n{grad_approx}\n\nDiff = \n{diff}\n\n '
            if logged:
                f.write(out)
            else:
                print(out)

        w = w - alpha * gradient

        # VRLO VAZAN KORAK: nakon sto je model azuriran, treba ga azurirati UNUTAR neuralne
        # mreze kako bi backward propagation algoritam radio sa novim vrednostima modela.
        neural_network.set_model(roll_vec_to_matrix_array(w, matrix_sizes))

    if logged:
        f.close()

    if plot:
        plt.plot(range(num_iter), loss_history)

        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        plt.legend(['Loss_train'])
        plt.title(f'Minimization of loss function via gradient descent\n[alpha={alpha}, lambda={lambda_param}]')

        plt.show()

    return loss_history, w


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_options, output_layer_size, eps_init=1e-1):
        self.input_layer_size = input_layer_size
        self.hidden_layers_num = len(hidden_layer_options)
        self.layers_num = self.hidden_layers_num + 2
        self.L = self.layers_num
        self.output_layer_size = output_layer_size
        self.layer_sizes = [input_layer_size] + hidden_layer_options + [output_layer_size]
        self.layer_indices = range(self.L)
        self.eps_init = eps_init

        if output_layer_size == 2:
            raise ValueError('Nekorektna velicina izlaznog sloja 2: za binarnu klasifikaciju'
                             'je 1, a za multiklasnu >= 3.')
        elif output_layer_size <= 0:
            raise ValueError(f'Nekorektna velicina izlaznog sloja {output_layer_size}:'
                             f'ocekivana 1 ili >= 3.')
        elif output_layer_size == 1:
            self.k_classes = 2
        else:
            self.k_classes = output_layer_size

        # NAPOMENA: velicina slojeva neuralne mreze (layer_sizes) se racuna kao broj jedinica
        # BEZ bias jedinice, ali prilikom postavljanja slojeva mreza ocekuje bias jedinice.
        # Dakle, svaki sloj je velicine za jedan vise, osim poslednjeg sloja.
        layers = [np.ones(input_layer_size + 1)]
        for i in range(self.hidden_layers_num):
            layers.append(np.ones(hidden_layer_options[i] + 1))

        layers.append(np.ones(output_layer_size))
        self.network = np.array(layers)

        # Cuvam dimenzije matrica Wij koje mapiraju slojeve j -> j + 1
        # s(j+1) x (s(j) + 1)
        mappers = {}
        for l in range(self.L - 1):
            rows = self.layer_sizes[l + 1]
            cols = self.layer_sizes[l] + 1

            mappers[l] = (rows, cols)

        self.layer_mapper_sizes = mappers

        # inicijalizacija modela
        ws = []
        for l in range(self.L - 1):
            w_l = 2 * self.eps_init * np.random.random(self.layer_mapper_sizes[l]) - self.eps_init
            ws.append(w_l)

        self.model = np.array(ws)
        self.model_trained = self.model

        # Analogno kao za slojeve, delte ocekuju bias jedinice, koje se NECE koristiti
        # u proracunima backpropagation algoritma. Prvi sloj mora da ima
        # podrazumevano za delte sve nule, jer ulazni podaci nemaju gresku.
        deltas = []
        for l in self.layer_indices:
            if l == self.L - 1:
                delta_l = np.zeros(self.layer_sizes[l])
            else:
                delta_l = np.zeros(self.layer_sizes[l] + 1)

            deltas.append(delta_l)

        self.deltas = np.array(deltas)

    def __str__(self):
        self.print_network()

    def __repr__(self):
        self.print_network()

    def __layer_index_check(self, layer_index):
        if layer_index < 0 or layer_index >= self.L:
            raise IndexError(f'Nekorektan indeks sloja neuralne mreze {layer_index}: '
                             f'dostupni indeksi 0-{self.L - 1}')

    def __set_layer_index_check(self, layer_index, set_vec):
        # poslednji sloj NEMA bias unit
        if layer_index != self.L - 1 and \
                set_vec.shape[0] != self.layer_sizes[layer_index] + 1:
            error_message = f'Nekorektna dimenzija vektora {set_vec.shape[0]} za sloj ' \
                f'{layer_index}: ocekivana {self.layer_sizes[layer_index] + 1}'

            raise ValueError(error_message)
        elif layer_index == self.L - 1 and \
                set_vec.shape[0] != self.layer_sizes[layer_index]:  # ovde je layer_index poslednji sloj
            error_message = f'Nekorektna dimenzija vektora {set_vec.shape[0]} za ' \
                f'izlazni sloj {layer_index}: ocekivana {self.layer_sizes[layer_index]}'

            raise ValueError(error_message)

    def set_layer(self, layer_index, units_vec):
        self.__layer_index_check(layer_index)
        self.__set_layer_index_check(layer_index, units_vec)

        self.network[layer_index] = units_vec

    def set_all_layers(self, all_layers):
        for l in range(self.L):
            self.set_layer(l, all_layers[l])

    def set_mapper(self, layer_index, mapper):
        self.__layer_index_check(layer_index)

        if mapper.shape != self.model[layer_index].shape:
            error_message = f'Nekorektna dimenzija matrice {mapper.shape[0]}x{mapper.shape[1]} ' \
                f'za preslikavanje sloja {layer_index} -> {layer_index + 1}: ocekivana ' \
                f'{self.model[layer_index].shape[0]}x{self.model[layer_index].shape[1]}'

            raise ValueError(error_message)

        self.model[layer_index] = mapper

    # Funkcija po analogiji za slojeve, radi doslednosti
    def set_all_mappers(self, all_mappers):
        for l in range(self.L - 1):
            self.set_mapper(l, all_mappers[l])

    def set_model(self, all_mappers):
        self.set_all_mappers(all_mappers)

    def set_delta(self, layer_index, delta_vec):
        self.__layer_index_check(layer_index)
        self.__set_layer_index_check(layer_index, delta_vec)

        self.deltas[layer_index] = delta_vec

    def set_all_deltas(self, all_deltas):
        for l in self.layer_indices:
            self.set_delta(l, all_deltas[l])

    def unroll_mapper(self, layer_index):
        self.__layer_index_check(layer_index)

        return self.model[layer_index].ravel()

    def unroll_model(self):
        return unroll_matrix_array(self.model)

    def initialize_model(self, model_init_bounds=(-1e-1, 1e-1)):
        a = model_init_bounds[0]
        b = model_init_bounds[1]

        global w_l
        ws = []

        for l in range(self.L - 1):
            w_l = (b - a) * np.random.random(self.layer_mapper_sizes[l]) + a
            ws.append(w_l)

        self.model = np.array(ws)

    def forward_propagation(self, input_layer_data):
        a_l = input_layer_data
        self.set_layer(0, a_l)

        for l in range(self.L - 1):
            z_lp1 = self.model[l].dot(a_l)  # z(l+1) = W(l)*a(l)
            a_l = g_sigmoid(z_lp1)  # aktivacija

            # Dodavanje bias jedinice u a_l vektor
            if l != self.L - 2:
                a_l = np.concatenate((np.array([1]), a_l))

            self.set_layer(l + 1, a_l)

    # Vrsi propagaciju na osnovu prosledjenog modela, a ne na osnovu internog modela.
    def propagate(self, neural_model, input_layer_data):
        a_l = input_layer_data
        self.set_layer(0, a_l)

        for l in range(self.L - 1):
            z_lp1 = neural_model[l].dot(a_l)
            a_l = g_sigmoid(z_lp1)

            # Dodavanje bias jedinice u a_l vektor
            if l != self.L - 2:
                a_l = np.concatenate((np.array([1]), a_l))

            self.set_layer(l + 1, a_l)

    # Vraca ono sto se nalazi u poslednjem sloju
    @property
    def output_layer(self):
        return self.network[self.L - 1]

    # Racuna se greska svakog neurona ponaosob
    def backward_propagation_deltas(self, y_data):
        delta_output = y_data - self.output_layer  # Proveriti abs ??
        self.set_delta(self.L - 1, delta_output)

        for l in range(self.L - 2, 0, -1):
            z_l = self.model[l - 1].dot(self.network[l - 1])  # z(l) = W(l-1)*a(l-1)
            g_prim_vec = np.concatenate((np.array([1]), g_sigmoid_derivative(z_l)))

            if l + 1 == self.L - 1:
                delta_lp1 = self.deltas[l + 1]  # ne postoji bias jedinica za poslednji sloj
            else:
                delta_lp1 = self.deltas[l + 1][1:]  # ignorisem bias jedinicu delta vektora

            delta_l = self.model[l].T.dot(delta_lp1) * g_prim_vec
            self.set_delta(l, delta_l)

        # delta_0 je uvek nula-vektor, postavljen jos prilikom inicijalizacije same mreze

    def backpropagation_depr(self, X_train, y_train, lambda_param=0):
        """ZASTARELA"""
        accs = []
        for l in range(self.L - 1):
            delta_acc_l = np.zeros(self.layer_mapper_sizes[l])
            accs.append(delta_acc_l)

        # Delta_l akumulatori za parcijalne izvode i inicijalni parcijalni izvodi
        delta_accumulators = np.array(accs)
        gradient = np.array(accs)
        loss_inner = 0

        N = X_train.shape[0]
        for i in range(N):
            self.forward_propagation(X_train[i])  # propagacija aktivacija unapred
            self.backward_propagation_deltas(y_train[i])  # propagacije greske unazad
            self.__accumulate_deltas(delta_accumulators)  # akumulacija greske

            y_predict = self.output_layer  # y_predict se zbog propagacije nalazi u poslednjem sloju
            loss_inner += loss_instance(y_train[i], y_predict)

        self.__set_partial_derivatives(gradient, delta_accumulators, N, lambda_param)
        loss = loss_inner / N + regularization_depr(self.model, N, lambda_param)

        return loss, unroll_matrix_array(gradient)

    def backpropagation(self, X_train, y_train, lambda_param=0):
        accs = []
        for l in range(self.L - 1):
            delta_acc_l = np.zeros(self.layer_mapper_sizes[l])
            accs.append(delta_acc_l)

        # Delta_l akumulatori za parcijalne izvode i inicijalni parcijalni izvodi
        delta_accumulators = np.array(accs)
        gradient = np.array(accs)
        loss_inner = 0

        N = X_train.shape[0]
        for i in range(N):
            self.forward_propagation(X_train[i])  # propagacija aktivacija unapred
            self.backward_propagation_deltas(y_train[i])  # propagacije greske unazad
            self.__accumulate_deltas(delta_accumulators)  # akumulacija greske

            y_predict = self.output_layer  # y_predict se zbog propagacije nalazi u poslednjem sloju
            loss_inner += loss_instance(y_train[i], y_predict)

        self.__set_gradient(gradient, delta_accumulators, N, lambda_param)
        loss = loss_inner / N + regularization(self.model, N, lambda_param)

        return loss, unroll_matrix_array(gradient)

    def __accumulate_deltas(self, delta_accumulators):
        for l in range(self.L - 1):
            a_l = self.network[l].reshape(-1, 1).transpose()

            if l + 1 == self.L - 1:
                delta_lp1 = self.deltas[l + 1].reshape(-1, 1)
            else:
                delta_lp1 = self.deltas[l + 1][1:].reshape(-1, 1)

            delta_accumulators[l] += delta_lp1.dot(a_l)

    def __set_partial_derivatives(self, partial_derivatives, delta_accumulators, N_set_size, lambda_param):
        """ZASTARELA, neefikasna"""
        for l in range(self.L - 1):
            for i in range(self.model[l].shape[0]):
                for j in range(self.model[l].shape[1]):
                    if j == 0:
                        partial_derivatives[l][i][j] = delta_accumulators[l][i][j] / N_set_size
                    else:
                        partial_derivatives[l][i][j] = delta_accumulators[l][i][j] / N_set_size + \
                                                       lambda_param * self.model[l][i][j]

    def __set_gradient(self, gradient, delta_acc, N, lambda_param):
        for l in range(self.L - 1):
            gradient[l] = delta_acc[l] / N + lambda_param * self.model[l]
            gradient[l][:, 0] -= (lambda_param * self.model[l][:, 0])

    def fit(self,
            X_data,
            y_data,
            alpha=0.01,
            num_iter=1000,
            lambda_param=0,
            model_init_bounds=(-1e-1, 1e-1),
            plot=False,
            gc_it=-1,
            gc_log_path=None):

        loss_history, model_trained = gradient_descent(X_data,
                                                       y_data,
                                                       self,
                                                       alpha=alpha,
                                                       num_iter=num_iter,
                                                       lambda_param=lambda_param,
                                                       model_init_bounds=model_init_bounds,
                                                       plot=plot,
                                                       gc_it=gc_it,
                                                       gc_log_path=gc_log_path)

        matrix_sizes = get_matrix_sizes(self.model)
        self.model_trained = roll_vec_to_matrix_array(model_trained, matrix_sizes)

        return Classifier(self.model_trained, lambda_learned=lambda_param)

    def predict(self, input_data):
        self.propagate(self.model_trained, input_data)

        return self.output_layer

    def print_layer(self, layer_index):
        self.__layer_index_check(layer_index)

        print(f'Layer {layer_index}:')
        print(self.network[layer_index])

    def print_network(self):
        print('~ Neural network ~')

        for i in range(self.network.shape[0]):
            print(f'Layer {i + 1}:')
            print(self.network[i])

    def print_layer_mapper_sizes(self):
        print('Dimenzije matrica modela koji mapiraju slojeve:')

        for k, v in self.layer_mapper_sizes.items():
            print(f'{k} -> {k + 1}: {v[0]} x {v[1]}')

    def print_mapper(self, layer_index):
        self.__layer_index_check(layer_index)

        print(f'W_{layer_index}: {layer_index} -> {layer_index + 1}, shape: {self.layer_mapper_sizes[layer_index]}')
        print(self.model[layer_index])

    def print_model(self):
        print('Model:')

        for l in range(self.L - 1):
            print(f'W_{l}: {l} -> {l + 1}, shape: {self.layer_mapper_sizes[l]}')
            print(self.model[l])

    def print_delta(self, layer_index):
        self.__layer_index_check(layer_index)

        print(self.deltas[layer_index])

    def print_deltas(self):
        print('Deltas:')

        for l in range(self.L):
            print(f'delta_{l}:')
            print(self.deltas[l])


class Classifier:
    def __init__(self, neural_model, lambda_learned=-1):
        self.model = neural_model
        self.lambda_learned = lambda_learned

    def predict(self, input_data):
        return hypothesis_neural(self.model, input_data)

    @staticmethod
    def prediction_info(y_predict, class_to_name_mapper=None):
        info = ''

        if type(y_predict) == np.ndarray:
            max_val = float('-inf')
            cls = 0

            for i, val in enumerate(y_predict):
                if max_val < val:
                    max_val = val
                    cls = i

            if class_to_name_mapper is not None:
                info = f'Prediction vector: {y_predict}\nMax prediction value: {max_val} (at ' \
                    f'position {cls})\nClass: {class_to_name_mapper[cls]}'
            else:
                info = f'Prediction vector: {y_predict}\nMax prediction value: {max_val}\nClass: {cls}'
        else:
            if y_predict < 0.5:
                cls = 0
            else:
                cls = 1

            if class_to_name_mapper is not None:
                info = f'Prediction: {y_predict}\nClass: {class_to_name_mapper[cls]}'
            else:
                info = f'Prediction: {y_predict}Class: {cls}'

        print(info)

    def loss(self, X_data, y_data, lambda_param=0):
        return loss_logistic(X_data, y_data, self.model, lambda_param)

    def print_model(self):
        print('Model:')

        for l in range(self.model.shape[0]):
            print(f'W_{l}: {l} -> {l + 1}')
            print(self.model[l])

