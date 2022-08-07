import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from desdeo_emo.othertools.ProbabilityWrong import Probability_wrong
import scipy
from scipy.stats import norm
import scipy.integrate as integrate

class ProbMOEAD_select(SelectionBase):
    """The MOEAD selection operator. 

    Parameters
    ----------
    pop : Population
        The population of individuals
    SF_type : str
        The scalarizing function employed to evaluate the solutions

    """
    def __init__(
        self, pop: Population, SF_type: str
    ):
	 # initialize
        self.SF_type = SF_type

    def do(self, pop: Population, vectors: ReferenceVectors, ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> List[int]:
        """Select the individuals that are kept in the neighborhood.

        Parameters
        ----------
        pop : Population
            The current population.
        vectors : ReferenceVectors
            Class instance containing reference vectors.
        ideal_point
            Ideal vector found so far
        current_neighborhood
            Neighborhood to be updated
        offspring_fx
            Offspring solution to be compared with the rest of the neighborhood

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        # Compute the value of the SF for each neighbor
        num_neighbors               = len(current_neighborhood)
        current_population          = pop.objectives[current_neighborhood,:]
        current_uncertainty          = pop.uncertainity[current_neighborhood,:]
        current_reference_vectors   = vectors.values[current_neighborhood,:]
        offspring_population        = np.array([offspring_fx]*num_neighbors)
        offspring_uncertainty       = np.array([offspring_unc]*num_neighbors)
        ideal_point_matrix          = np.array([ideal_point]*num_neighbors)
        theta_adaptive_matrix       = np.array([theta_adaptive]*num_neighbors)
        n_samples = 1000
        pwrong_current = Probability_wrong(mean_values=current_population, stddev_values=current_uncertainty, n_samples=n_samples)
        pwrong_current.vect_sample_f()

        pwrong_offspring = Probability_wrong(mean_values=offspring_population.reshape(-1,pop.problem.n_of_objectives), stddev_values=offspring_uncertainty.reshape(-1,pop.problem.n_of_objectives), n_samples=n_samples)
        pwrong_offspring.vect_sample_f()

        values_SF_current = self._evaluate_SF(current_population, current_reference_vectors, ideal_point_matrix, pwrong_current, theta_adaptive_matrix)
        values_SF_offspring = self._evaluate_SF(offspring_population, current_reference_vectors, ideal_point_matrix, pwrong_offspring, theta_adaptive_matrix)

        ##### KDE here and then compute probability
        pwrong_current.pdf_list = {}
        pwrong_current.ecdf_list = {}
        pwrong_offspring.pdf_list = {}
        pwrong_offspring.ecdf_list = {}
        values_SF_offspring_temp = np.asarray([values_SF_offspring])
        values_SF_current_temp = np.asarray([values_SF_current])
        pwrong_offspring.compute_pdf(values_SF_offspring_temp.reshape(num_neighbors,1,n_samples))
        pwrong_current.compute_pdf(values_SF_current_temp.reshape(num_neighbors,1,n_samples))
        probabilities = np.zeros(num_neighbors)
        for i in range(num_neighbors):
            # Using integral of KDE pdf
            #probabilities[i]=pwrong_current.compute_probability_wrong_PBI(pwrong_offspring, index=i)
            #print("P_wrong_integral:",probabilities[i])
            # cheaper MC samples comparison
            #probabilities[i]=pwrong_current.compute_probability_wrong_MC(values_SF_current[i], values_SF_offspring[i])
            #print("P_wrong_MC:",probabilities[i])
            # closed form TCH computation
            probabilities[i]= self.compute_probability_wrong_TCH(current_population[i],current_uncertainty[i],offspring_fx.reshape(-1), offspring_unc.reshape(-1) ,current_reference_vectors[i], ideal_point)
            #print("P_wrong_analytical:",probabilities[i])
        # Compare the offspring with the individuals in the neighborhood 
        # and replace the ones which are outperformed by it if P_{wrong}>0.5
        selection = np.where(probabilities>0.5)[0]
        #print("Selection:",selection)
        return current_neighborhood[selection]

    def get_pdf_g_tcheby(self, x, w, z, mu_f, sigma_f):
        g=x
        m=w*(mu_f-z)
        s=w*sigma_f
        g_m_s = (g-m)/s
        pdf_i = norm.pdf(g_m_s)
        cdf_i = norm.cdf(g_m_s)
        s=s+0.00000000001
        cdf_i = cdf_i + 0.00000000001 
        prod_cdf_g = np.prod(cdf_i)
        sigma_term = np.sum((pdf_i/cdf_i)/s)
        pdf_g = sigma_term * prod_cdf_g
        #print(pdf_g)
        return pdf_g
    
    def compute_cdf_TCH(self, g, w, z, mu_f, sigma_f):
        return integrate.quad(self.get_pdf_g_tcheby,0, g, args=(w, z, mu_f, sigma_f))[0]

    def compute_inner_product(self, x, mu_current, unc_current, mu_off, unc_off, ref_v, ideal):
        inner_product = self.get_pdf_g_tcheby(x, ref_v, ideal, mu_current, unc_current) * self.compute_cdf_TCH(x, ref_v, ideal, mu_off, unc_off)
        #print(inner_product)
        return inner_product

    def compute_probability_wrong_TCH(self, mu_current, unc_current, mu_off, unc_off, ref_v, ideal):
        p_wrong = integrate.quad(self.compute_inner_product,0, np.inf, args=(mu_current, unc_current, mu_off, unc_off, ref_v, ideal))[0]
        return p_wrong

    def tchebycheff(self, objective_values, weights, ideal_point, pwrong_f_samples):
        #feval   = np.abs(objective_values - ideal_point) * weights
        feval   = np.abs(np.transpose(pwrong_f_samples) - np.tile(ideal_point,(1000,1))) * np.tile(weights,(1000,1))
        max_fun = np.max(feval, axis=1)
        
        return max_fun

    def weighted_sum(self, objective_values, weights, pwrong_f_samples):
        #feval   = np.sum(objective_values * weights)
        feval   = np.sum(np.transpose(pwrong_f_samples) * np.tile(weights,(1000,1)), axis=1)
        return feval
    
    def calc_m(self, mu_a, mu_b, sigma_b, weights):
        return (np.sum(mu_a*weights)-np.sum(mu_b*weights))/np.sqrt(np.sum((weights**2)*(sigma_b**2)))
        #return (np.sum(mu_a*weights)-np.sum(mu_b*weights))/np.sum((weights)*(sigma_b))

    def calc_s(self, sigma_a, sigma_b, weights):
        return np.sqrt(np.sum((weights**2)*(sigma_a**2)))/np.sqrt(np.sum((weights**2)*(sigma_b**2)))
        #return np.sum((weights)*(sigma_a))/np.sum((weights)*(sigma_b))

    def compute_probability_wrong_WS_analytic(self, parent_mean, parent_unc, offspring_mean, offspring_unc, weights):
        m= self.calc_m(parent_mean, offspring_mean, offspring_unc, weights)
        s = self.calc_s(parent_unc, offspring_unc, weights)
        #print("m :",m)
        #print("s :",s)
        pwrong_A_B = 0.5+0.5*scipy.special.erf(m/np.sqrt(2+2*s**2))
        #print("P_wrong_erf:",pwrong_A_B)
        # erf approximation using tanh
        #pwrong_A_B = 0.5*(1+np.tanh(m/(0.8*np.sqrt(2+2*s**2))))
        #print("P_wrong_tanh:",pwrong_A_B)
        return pwrong_A_B

    def pbi(self, objective_values, weights, ideal_point, pwrong_f_samples, theta):

        norm_weights    = np.linalg.norm(weights)
        weights         = weights/norm_weights
        fx_a            = pwrong_f_samples - ideal_point.reshape(-1,1)
        d1               = np.sum(np.transpose(fx_a)* np.tile(weights,(1000,1)), axis=1)
        fx_b             = np.transpose(pwrong_f_samples) - (np.tile(ideal_point,(1000,1)) + np.reshape(d1,(-1,1)) * np.tile(weights,(1000,1)))
        d2               = np.linalg.norm(fx_b, axis=1)

        fvalue          = d1 + theta * d2

        return fvalue


    def _evaluate_SF(self, neighborhood, weights, ideal_point, pwrong, theta_adaptive):
        if self.SF_type == "TCH":
            SF_values = np.array(list(map(self.tchebycheff, neighborhood, weights, ideal_point, pwrong.f_samples)))
            return SF_values
        elif self.SF_type == "PBI":
            SF_values = np.array(list(map(self.pbi, neighborhood, weights, ideal_point, pwrong.f_samples, theta_adaptive)))
            return SF_values
        elif self.SF_type == "WS":
            SF_values = np.array(list(map(self.weighted_sum, neighborhood, weights, pwrong.f_samples)))
            return SF_values
        else:
            return []



    

    

    
    

