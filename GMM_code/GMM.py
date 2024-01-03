import numpy as np
from scipy import random
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None, labeled_data_weight=None):
        '''
        Define a model with known number of clusters and dimensions.
        input:
            - k: Number of Gaussian clusters
            - dim: Dimension 
            - init_mu: initial value of mean of clusters (k, dim)
                       (default) random from uniform[-10, 10]
            - init_sigma: initial value of covariance matrix of clusters (k, dim, dim)
                          (default) Identity matrix for each cluster
            - init_pi: initial value of cluster weights (k,)
                       (default) equal value to all cluster i.e. 1/k
            - colors: Color valu for plotting each cluster (k, 3)
                      (default) random from uniform[0, 1]
        '''
        self.k = k
        self.dim = dim
        if(init_mu is None):
            init_mu = random.rand(k, dim)*20 - 10
        if labeled_data_weight is not None:
            assert labeled_data_weight > 1
            # assert labeled_data_weight == int(labeled_data_weight)
            
        self.labeled_data_weight = labeled_data_weight
        self.mu = init_mu
        if(init_sigma is None):
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        if(init_pi is None):
            init_pi = np.ones(self.k)/self.k
        self.pi = init_pi
        if(colors is None):
            colors = random.rand(k, 3)
        self.colors = colors

    def init_params_with_kmeans(self, X):
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.k, random_state=0, n_init=10)
        kmeans.fit(X)

        # Initialize GMM means with k-means centroids
        self.mu = kmeans.cluster_centers_

        # Initialize covariance matrices and weights
        self.sigma = np.array([np.eye(self.dim) for _ in range(self.k)])  # or use variances from clusters
        labels = kmeans.labels_
        self.pi = np.array([np.sum(labels == i) for i in range(self.k)]) / len(X)
    
    def init_em(self, X, observed_idxs=None, observed_labels=None):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.num_points = X.shape[0]
        if observed_idxs is None:
            self.observed_idxs = []
            assert observed_labels is None
            self.observed_labels = []
            self.unobserved_idxs = list(range(self.num_points))
        else:
            assert observed_labels is not None
            assert len(observed_idxs) == len(observed_labels)
            self.observed_idxs = observed_idxs
            self.observed_labels = observed_labels
            self.unobserved_idxs = list(set(range(self.num_points)) - set(observed_idxs))
        self.z = np.zeros((self.num_points, self.k))
    
    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.k):
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        self.z /= self.z.sum(axis=1, keepdims=True)
        if len(self.observed_idxs) > 0:
            self.z[self.observed_idxs, :] = 0
            for i, idx in enumerate(self.observed_idxs):
                self.z[idx, self.observed_labels[i]] = 1

    def get_params(self):
        '''
        Return current parameters.
        '''
        params = {'mean':[], 'cov':[], 'pi':[]}
        for i in range(self.k):
            params['mean'].append(self.mu[i])
            params['cov'].append(self.sigma[i])
            params['pi'].append(self.pi[i])
        return params
    
    def fit(self, num_iters, tol=1e-4, verbose=False):
        log_likelihood = [self.log_likelihood(self.data)] # warning: this does not take labeled data into account. 
        for e in range(num_iters):
            # E-step
            self.e_step()
            # M-step
            self.m_step()
            # Computing log-likelihood
            log_likelihood.append(self.log_likelihood(self.data))
            if verbose:
                print("Iteration: {}, log-likelihood: {:.4f}".format(e+1, log_likelihood[-1]))
            if np.abs(log_likelihood[-1] - log_likelihood[-2]) < tol:
                if verbose:
                    print("Converged")
                    break
    
    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        # print("z", self.z)
        # if len(self.observed_idxs) == 0:
        #     z_to_use = self.z
        #     X_to_use = self.data
        #     num_points_to_use = self.num_points
        # else:
        #     # concatenate the unobserved data to labeled_data_weight times of labeled data
        #     z_to_use = np.concatenate([self.z[self.unobserved_idxs, :], 
        #                                np.repeat(self.z[self.observed_idxs, :], self.labeled_data_weight, axis=0)], axis=0)
        #     X_to_use = np.concatenate([self.data[self.unobserved_idxs, :],
        #                                 np.repeat(self.data[self.observed_idxs, :], self.labeled_data_weight, axis=0)], axis=0)
        #     num_points_to_use = z_to_use.shape[0]
        #     assert num_points_to_use == X_to_use.shape[0] == len(self.unobserved_idxs) + len(self.observed_idxs) * self.labeled_data_weight
        # # 
        # sum_z = z_to_use.sum(axis=0)
        # self.pi = sum_z / num_points_to_use
        # self.mu = np.matmul(z_to_use.T, X_to_use)
        # self.mu /= sum_z[:, None]
        # for i in range(self.k):
        #     # EP DID NOT WRITE THIS CODE NOR DO I UNDERSTAND IT BUT IT SEEMS TO NOT PERFECTLY AGREE WITH COVARIANCE MATRIX, POSSIBLY BECAUSE OF N vs N-1 ISSUES. IT'S VERY CLOSE THOUGH. 
        #     j = np.expand_dims(X_to_use, axis=1) - self.mu[i]
        #     s = np.matmul(j.transpose([0, 2, 1]), j)
        #     self.sigma[i] = np.matmul(s.transpose(1, 2, 0), z_to_use[:, i]) #
        #     adjusted_sum = sum_z[i] - 1 if sum_z[i] > 1 else 1
        #     # self.sigma[i] /= sum_z[i]
        #     self.sigma[i] /= adjusted_sum
        
        ###### NEW CODE STARTS HERE ######
        # Create effective_z for upweighting labeled data
        effective_z = np.copy(self.z)
        if self.labeled_data_weight is not None and len(self.observed_idxs) > 0:
            effective_z[self.observed_idxs, :] *= self.labeled_data_weight

        # Calculate the sum of responsibilities for each cluster
        sum_z = np.sum(effective_z, axis=0)

        # Update mixture weights (pi)
        self.pi = sum_z / np.sum(sum_z)
        # print("pi", self.pi, pi, "\n")

        mu = np.copy(self.mu)
        for i in range(self.k):
            weighted_data = self.data.T * effective_z[:, i]
            self.mu[i] = np.sum(weighted_data, axis=1) / sum_z[i]

        # Update means (mu)
        # weighted_data = self.data * effective_z[:, :, np.newaxis]
        # mu = np.sum(weighted_data, axis=0) / sum_z[:, np.newaxis]

        # print("mu", self.mu, mu, "\n")

    #    sigma = np.copy(self.sigma)

        # Update covariance matrices (sigma)
        for i in range(self.k):
            diff = np.expand_dims(self.data, axis=1) - mu[i]
    
            # Compute the weighted sum of outer products
            # effective_z[:, i][:, np.newaxis, np.newaxis] reshapes the weights for broadcasting
            weighted_outer_product = np.sum(diff * diff.transpose(0, 2, 1) * effective_z[:, i][:, np.newaxis, np.newaxis], axis=0)
            
            # Normalize by the effective sum of weights (adjusted for N-1)
            adjusted_sum = sum_z[i] - 1 if sum_z[i] > 1 else 1
            self.sigma[i] = weighted_outer_product / adjusted_sum + np.eye(self.dim) * 1e-6
        
        # print("sigma", np.sum(np.abs(sigma-self.sigma)))

        
            
    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
        # ll = []
        ll = 0
        # for d in X:
        #     tot = 0
        #     for i in range(self.k):
        #         tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
        #     ll.append(np.log(tot))
        # return np.sum(ll)
        for n in range(len(X)):
            if n in self.observed_idxs:
                # Find the label for the current observed index
                label_index = np.where(self.observed_idxs == n)[0][0]
                label = self.observed_labels[label_index]
                ll += np.log(multivariate_normal.pdf(X[n], mean=self.mu[label], cov=self.sigma[label]) + 1e-9)
            else:
                # For unlabeled data, sum over all components
                tot = 0
                for i in range(self.k):
                    tot += self.pi[i] * multivariate_normal.pdf(X[n], mean=self.mu[i], cov=self.sigma[i])
                ll += np.log(tot + 1e-9)  # Adding a small constant to avoid log(0)
        return ll
    
    def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        '''
        Utility function to plot one Gaussian from mean and covariance.
        '''
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
        '''
        Function to draw the Gaussians.
        Note: Only for two-dimensionl dataset
        '''
        if(self.dim != 2):
            print("Drawing available only for 2D case.")
            return
        for i in range(self.k):
            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)