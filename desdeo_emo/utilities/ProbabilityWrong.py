import numpy as np
import scipy.integrate as integrate
#from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KernelDensity
from scipy.stats import truncnorm
import multiprocessing as mp
import itertools
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import rc
from datetime import datetime
from scipy.stats import norm

warnings.filterwarnings("ignore")

rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})

rx = np.zeros((3,2,2))
operator = 'APD'
n_samples= 1000

class Probability_wrong:
    """Class for computing the probability of wrong selection of a distribution"""
    def __init__(self, mean_values=None, stddev_values=None, n_samples=1000,p=None):
        self.mean_values = mean_values
        self.stddev_values = stddev_values
        self.n_samples = n_samples
        self.f_samples = None
        self.size_f = np.shape(mean_values)[0]
        self.num_objectives = np.shape(mean_values)[1]
        self.pdf_list = {}
        self.ecdf_list = {}
        self.pdf_grids = None
        self.cdf_grids = None
        self.support_grids = None
        self.pdf_cdf = None
        self.rank_prob_wrong = None
        self.lower_bound = None
        self.upper_bound = None
        self.apd_mean_samples = {}
        self.apd_sigma_samples = {}
        self.mean_samples = None
        self.sigma_samples = None
        self.size_rows = None
        self.size_cols = None
        self.p = p
        self.apd_pdf_list = {}
        self.apd_ecdf_list = {}
        self.parallel_list = {}

    def vect_sample_f(self):
        for i in range(self.size_f):
            f_temp = None
            for j in range(self.num_objectives):
                #sample = truncnorm.rvs(-1.96, 1.96, loc=self.mean_values[i,j], scale=self.stddev_values[i,j], size=self.n_samples)
                sample = truncnorm.rvs(-3, 3, loc=self.mean_values[i, j], scale=self.stddev_values[i, j],
                                       size=self.n_samples)
                #sample = np.random.normal(self.mean_values[i,j], self.stddev_values[i,j], self.n_samples)
                sample = np.reshape(sample, (1,1,self.n_samples))
                if f_temp is None:
                    f_temp = sample
                else:
                    f_temp = np.hstack((f_temp, sample))
            if self.f_samples is None:
                self.f_samples = f_temp
            else:
                self.f_samples = np.vstack((self.f_samples, f_temp))

    def compute_pdf(self, samples=None, bw=None):
        if samples is None:
            samples = self.f_samples
        self.mean_samples = np.mean(samples, axis=2)
        self.sigma_samples = np.std(samples, axis=2)
        self.size_rows = np.shape(samples)[0]
        self.size_cols = np.shape(samples)[1]
        self.lower_bound = np.min(samples, axis=2)
        self.upper_bound = np.max(samples, axis=2)
        if bw is None:
            bw = 1.06*self.sigma_samples/np.power(self.n_samples, (1/5))
        else:
            bw = np.ones((self.size_rows,self.size_cols))*bw
        bw[bw <= 0.0] = 0.000000000000001

        for i in range(self.size_rows):
            pdf_temp = []
            ecdf_temp = []
            for j in range(self.size_cols):
                sample_temp = samples[i, j, :]
                kde = KernelDensity(kernel='gaussian', bandwidth=bw[i, j]).fit(np.reshape(sample_temp, (-1, 1)))
                pdf_temp.append(kde)
                #sample_temp = self.pdf_predict(sample, kde)
                ecdf = ECDF(sample_temp)
                ecdf_temp.append(ecdf)
            self.pdf_list[str(i)]=pdf_temp
            self.ecdf_list[str(i)]=ecdf_temp

    def pdf_predict(self, x, pdf1, mu_B=None):
        pdf_vals = None
        if mu_B is None:
            pdf_vals = np.exp(pdf1.score_samples(np.reshape(x, (-1, 1))))
        else:
            return np.exp(pdf1.score_samples(np.reshape(mu_B-x, (-1, 1))))
        pdf_vals[np.where(x < 0)] = 0
        return pdf_vals

    def find_cdf(self, pdf, mu_B, lb_B, ub_B, mu):
        pdf1 = pdf
        return integrate.quad(self.pdf_predict, mu_B - mu, np.inf, args=(pdf1, mu_B))[0]



    def prob_mult(self, x, pdf_A, cdf_B):
        zz = self.pdf_predict(x, pdf_A)
        #print(zz)

        #kk = self.find_cdf(pdf_B, mu_B, lb_B, ub_B, mu=x)
        kk = cdf_B(x)
        #print("cdf")
        #print(kk)
        return zz * kk

    def compute_probability_wrong(self, pdf_A, pdf_B, mu_B):
        prob_wrong = integrate.quad(self.prob_mult2, -np.inf, np.inf, args=(pdf_A, pdf_B, mu_B))

        #prob_wrong = integrate.quad(self.prob_mult, self.lower_bound-1, self.upper_bound+1, args=(pdf_A, pdf_B, mu_B))
        #print(mu_B)
        print(prob_wrong[0])
        return prob_wrong[0]

    def find_cdf2(self, pdf, mu_B, mu):
        return integrate.quad(self.pdf_predict, mu_B - mu, np.inf, args=(pdf, mu_B))[0]

    def prob_mult2(self, x, pdf_A, pdf_B, mu_B):
        zz = self.pdf_predict(x, pdf_A)
        #print(zz)
        kk = self.find_cdf2(pdf_B, mu_B, mu=x)
        #print("cdf")
        #print(kk)
        return zz * kk

    def compute_probability_wrong2(self, i, j, k):
        pdf_A, pdf_B, mu_B = \
            self.pdf_list[str(i)][j], self.pdf_list[str(i)][k], self.mean_samples[i][k]
        prob_wrong = integrate.quad(self.prob_mult, -np.inf, np.inf, args=(pdf_A, pdf_B, mu_B))
        print(prob_wrong[0])
        return prob_wrong[0]

    def compute_probability_wrong_fast(self, i, j, k):
        pdf_A, pdf_B, mu_A, mu_B, sigma_A, sigma_B = \
            self.pdf_list[str(i)][j], \
            self.pdf_list[str(i)][k], \
            self.mean_samples[i][j], \
            self.mean_samples[i][k], \
            self.sigma_samples[i][j], \
            self.sigma_samples[i][k]
        cdf_B = self.ecdf_list[str(i)][k]
        lb_B = self.lower_bound[i, k]
        ub_B = self.upper_bound[i, k]
        lb_A = mu_A - 2.6 * sigma_A
        ub_A = mu_A + 2.6 * sigma_A
        if k<j:
            return -1
        elif j==k:
            return 0.5
        elif (mu_A+3*sigma_A < mu_B-3*sigma_B) and (mu_A < mu_B):
            return 0
        elif (mu_A-3*sigma_A > mu_B+3*sigma_B) and (mu_A > mu_B):
            return 1
        prob_mult_vect = np.vectorize(self.prob_mult)
        prob_wrong = integrate.quad(prob_mult_vect, 0, np.inf, args=(pdf_A, cdf_B))
        return prob_wrong[0]

    def compute_probability_wrong_blaze(self, i, j, k):
        pdf_A, pdf_B, mu_A, mu_B, sigma_A, sigma_B = \
            self.apd_pdf_list[str(i)]['0'][j], \
            self.apd_pdf_list[str(i)]['0'][k], \
            self.apd_mean_samples[str(i)][0][j], \
            self.apd_mean_samples[str(i)][0][k], \
            self.apd_sigma_samples[str(i)][0][j], \
            self.apd_sigma_samples[str(i)][0][k]
        cdf_B = self.apd_ecdf_list[str(i)]['0'][k]

        if k<j:
            return -1
        elif j==k:
            return 0.5
        elif (mu_A+3*sigma_A < mu_B-3*sigma_B) and (mu_A < mu_B):
            return 0
        elif (mu_A-3*sigma_A > mu_B+3*sigma_B) and (mu_A > mu_B):
            return 1

        prob_mult_vect = np.vectorize(self.prob_mult)

        prob_wrong = integrate.quad(prob_mult_vect, 0, np.inf, args=(pdf_A, cdf_B))
        return prob_wrong[0]

    def compute_probability_wrong_superfast(self, i, j, k):
        lb_A = np.min(self.support_grids[i,j])
        lb_B = np.min(self.support_grids[i,k])
        ub_A = np.max(self.support_grids[i,j])
        ub_B = np.max(self.support_grids[i,k])
        if lb_A > ub_B:
            return 1
        elif ub_A < lb_B:
            return 0
        elif j==k:
            return 0.5
        else:
            lb_int = max(lb_A, lb_B)
            ub_int = min(ub_A, ub_B)


        integrate.simps(self.pdf_cdf[i,j],)

    def fun_wrapper(self, indices):
        return self.compute_probability_wrong(*indices)
        #return self.compute_probability_wrong_fast(*indices)

    def fun_wrapper2(self, indices):
        return self.compute_probability_wrong_fast(*indices)

    def fun_wrapper3(self, indices):
        return self.compute_probability_wrong_blaze(*indices)

    def compute_rank(self):
        dim1 = self.size_rows
        dim2 = self.size_cols
        dim3 = self.size_cols
        self.rank_prob_wrong = np.zeros((dim1,dim2))
        for i in range(self.size_rows):
            for j in range(self.size_cols):
                temp_rank = 0
                for k in range(self.size_cols):
                    print(i)
                    print(j)
                    print(k)
                    temp_rank += self.compute_probability_wrong(
                                                            self.pdf_list[str(i)][j],
                                                            self.pdf_list[str(i)][k],
                                                            self.mean_samples[i][k])
                self.rank_prob_wrong[i, j] = temp_rank - 0.5

    def compute_rank_vectorized(self):
        vect_prob = np.vectorize(self.compute_probability_wrong, otypes=[np.float], cache=False)
        for i in range(self.size_f):
            for j in range(self.num_objectives):
                print(i)
                print(j)
                temp_rank = np.asarray(vect_prob(self.pdf_list[str(i)][j],
                                                            self.pdf_list[str(i)][:],
                                                            self.mean_samples[i][:]))
                temp_rank = np.sum(temp_rank)
                print(temp_rank)
                self.rank_prob_wrong[i, j] = temp_rank - 0.5

    def compute_rank_vectorized2(self):
        #p = mp.Pool(mp.cpu_count())
        p = mp.Pool(1)
        dim1 = self.size_rows
        dim2 = self.size_cols
        dim3 = self.size_cols
        self.rank_prob_wrong = np.zeros((dim1,dim2,dim3))
        input = ((i, j, k) for i, j, k in
                 itertools.product(range(dim1), range(dim2), range(dim3)))
        results = p.map(self.fun_wrapper2, input)
        p.close()
        p.join()
        results = np.asarray(results)
        results = np.reshape(results,(dim1,dim2,dim3))
        for i in range(dim1):
            for j in range(dim2):
                for k in range(dim3):
                    if k<j:
                        results[i,j,k]=1-results[i,k,j]
        self.rank_prob_wrong = np.sum(results, axis=2)-0.5

    def compute_rank_vectorized_apd(self, apd_list, indiv_index_list):
        p = mp.Pool(mp.cpu_count())
        dim1 = len(apd_list)
        for i in apd_list:
            self.compute_pdf(apd_list[i])
            self.apd_pdf_list[i] = self.pdf_list.copy()
            self.apd_ecdf_list[i] = self.ecdf_list.copy()
            self.apd_mean_samples[i] = self.mean_samples
            self.apd_sigma_samples[i] = self.sigma_samples
        #print("All PDF/CDF computed ...")

        count = 0
        for i in apd_list:
            for j in range(np.shape((apd_list[i]))[1]):
                for k in range(np.shape((apd_list[i]))[1]):

                    self.parallel_list[str(count)]=[int(i),j,k]
                    count += 1

        input = ((self.parallel_list[i][0], self.parallel_list[i][1], self.parallel_list[i][2]) for i in self.parallel_list)
        #print("Computing probabilities!")
        results=p.map(self.fun_wrapper3, input)
        print("Done!")
        p.close()
        p.join()
        prob_matrix={}
        results = np.asarray(results)
        count = 0
        for i in apd_list:
            prob_temp=np.zeros((np.shape((apd_list[i]))[1],np.shape((apd_list[i]))[1]))
            for j in range(np.shape((apd_list[i]))[1]):
                for k in range(np.shape((apd_list[i]))[1]):
                    if j > k:
                        prob_temp[j][k] = 1 - prob_temp[k][j]
                    else:
                        prob_temp[j][k] = results[count]
                    count += 1
            prob_matrix[i] = prob_temp

        rank_apd_matrix = {}
        for i in prob_matrix:
            rank_apd_matrix[i] = np.sum(prob_matrix[i], axis=1)-0.5
        selection = []
        for i in rank_apd_matrix:
            selection = np.append(selection,indiv_index_list[i][np.argmin(rank_apd_matrix[i])])

        return selection.astype(int)

    def plt_density(self, samples):
        #X_plot = np.linspace(-1, 20, 3000)
        #plt.rcParams["text.usetex"] = True

        for i in range(self.size_rows):
            X_plot = np.linspace(0, np.max(samples[i,:,:]), 1000)
            for j in range(self.size_cols):
                #plt.rcParams["text.usetex"] = True
                fig, ax = plt.subplots()
                fig.set_size_inches(4, 3.5)
                y = self.pdf_predict(X_plot, self.pdf_list[str(i)][j])
                y_2 = self.ecdf_list[str(i)][j](X_plot)
                #print(i)
                #print(j)
                #plt.xlim(1.5,4)
                ax.set_xlabel(operator)
                ax.set_ylabel('Probability density',color='r')

                #ax.plot(samples[i,j,:], (np.random.rand(samples.shape[2])*-0.02)-0.02, 'g+', ms=10, label='APD samples')
                ax.plot(samples[i, j, :], (np.random.rand(samples.shape[2])*0.02)+0.02, 'g+', ms=10,
                        label=operator+' samples')
                ax.plot(X_plot, y, label = 'Estimated PDF of '+operator,color='r')
                ax.hist(samples[i,j,:],30,alpha=0.5,density=True, label=('Histogram of '+operator+' samples'))
                ax.tick_params(axis='y',labelcolor='r')

                ax2=ax.twinx()

                ax2.set_ylabel('Cumulative density',color='b')
                ax2.plot(X_plot, y_2, label='Empirical CDF of  '+operator,color='b')
                ax2.tick_params(axis='y', labelcolor='b')
                #ax.legend()
                #ax.legend(loc='lower left',bbox_to_anchor=(0,1),ncol=1, fancybox=True)
                #ax2.legend()
                fig.tight_layout()

                fig.savefig('./Plots/'+operator+'_dist_'+str(int(datetime.timestamp(datetime.now())*1000))+'.pdf')
                #plt.show()
                print('Plot!')
    
    
    def compute_probability_wrong_PBI(self, pwrong_offspring, index):
        pdf_A, mu_A, mu_B, sigma_A, sigma_B = \
            self.pdf_list[str(index)][0], \
            self.mean_samples[index][0], \
            pwrong_offspring.mean_samples[index][0], \
            self.sigma_samples[index][0], \
            pwrong_offspring.sigma_samples[index][0]
        cdf_B = pwrong_offspring.ecdf_list[str(index)][0]
        max_range = max(mu_A+3*sigma_A,mu_B+3*sigma_B)
        if (mu_A+3*sigma_A < mu_B-3*sigma_B) and (mu_A < mu_B):
            return 0
        elif (mu_A-3*sigma_A > mu_B+3*sigma_B) and (mu_A > mu_B):
            return 1
        prob_mult_vect = np.vectorize(self.prob_mult)
        #prob_wrong = integrate.quad(prob_mult_vect, 0, np.inf, args=(pdf_A, cdf_B))
        prob_wrong = integrate.quad(prob_mult_vect, 0, max_range, args=(pdf_A, cdf_B))
        return prob_wrong[0]

    def compute_probability_wrong_MC(self, samples_A, samples_B):
        # Compute P_{wrong}(A>B)
        a_final=np.tile(samples_A,(self.n_samples,1))
        b_final=np.transpose(np.tile(samples_B,(self.n_samples,1)))
        return(np.sum(a_final>b_final)/(self.n_samples**2))
    

    def get_pdf_g_tcheby(self, x, w, z, mu_f, sigma_f):
        m=w*(mu_f-z)
        s=w*sigma_f
        g_m_s = (g-m)/s
        pdf_i = norm.pdf(g_m_s)
        cdf_i = norm.cdf(g_m_s)
        prod_cdf_g = np.prod(cdf_i)
        sigma_term = np.sum((pdf_i/cdf_i)/s)
        pdf_g = sigma_term * prod_cdf_g
        return pdf_g
    
    def compute_cdf_TCH(self, g, w, z, mu_f, sigma_f):
        return integrate.quad(self.get_pdf_g_tcheby,0, g, args=(w, z, mu_f, sigma_f))

    def compute_inner_product(self, x, mu_current, unc_current, mu_off, unc_off, ref_v, ideal):
        return self.get_pdf_g_tcheby(x, ref_v, ideal, mu_current, unc_current) * self.compute_cdf_TCH(x, ref_v, ideal, mu_off, unc_off)

    def compute_probability_wrong_TCH(self, mu_current, unc_current, mu_off, unc_off, ref_v, ideal):
        p_wrong = integrate.quad(self.get_pdf_g_tcheby,0, np.inf, args=(mu_current, unc_current, mu_off, unc_off, ref_v, ideal))
        return p_wrong
