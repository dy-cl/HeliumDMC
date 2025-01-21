#include <array>
#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>

#define numWalkersTarget 1000
#define numSteps 30000
#define dtau 0.005

// Walker object
class Walker
{
public:
    Eigen::Vector3d pos1; // Electron 1 position  
    Eigen::Vector3d pos2; // Electron 2 position 
};

// Guide function
double trialWavefunctionSquared(Walker walker, double alpha)
{   
    double r12 = (walker.pos1 - walker.pos2).norm();
    return std::exp(-4.0 * walker.pos1.norm()) 
          *std::exp(-4.0 * walker.pos2.norm()) 
          *std::exp(r12 / (1.0 + alpha * r12));
}

double localEnergy(Walker walker, double alpha)
{
    double r12 = (walker.pos1 - walker.pos2).norm();
    double term1 = (walker.pos1.normalized() - walker.pos2.normalized()).dot(walker.pos1 - walker.pos2) * (1 / (r12 * (1 + alpha*r12) * (1 + alpha*r12)));
    double term2 = 1 / (r12 * (1 + alpha*r12) * (1 + alpha*r12) * (1 + alpha*r12));
    double term3 = 1 / (4 * (1 + alpha*r12) * (1 + alpha*r12) * (1 + alpha*r12) * (1 + alpha*r12));
    double EL = -4 + term1 - term2 - term3 + 1/r12;

    //std::cout << EL << std::endl;
 
    return EL;
}

// Initialise all walkers at same position in 6D configuration space
void initialiseWalkers(std::vector<Walker>& walkers)
{  
    // Assign all walkers the same position
    for (Walker& walker : walkers)
    {
        walker.pos1 = Eigen::Vector3d(1,1,1);
        walker.pos2 = Eigen::Vector3d(-1,-1,-1);
    }
}

// Displaces walkers as according to Fokker-Planck force
void shiftFokkerPlanck(Walker& walker, double alpha)
{   
    // Calculate Fokker-Planck Forces
    double r12 = (walker.pos1 - walker.pos2).norm();
    Eigen::Vector3d F1 = -4*walker.pos1.normalized() + (walker.pos1 - walker.pos2) / (r12 * (1 + alpha * r12) * (1 + alpha * r12));
    Eigen::Vector3d F2 = -4*walker.pos2.normalized() + (walker.pos2 - walker.pos1) / (r12 * (1 + alpha * r12) * (1 + alpha * r12));

    // Shift walker
    walker.pos1 += F1*(dtau/2);
    walker.pos2 += F2*(dtau/2);
}

// Displace walkers by a random variable drawn from gaussian distribution
void shiftEtaGaussian(Walker& walker, double gamma, std::mt19937& gen)
{
    std::normal_distribution<double> dist(0.0, gamma);
    
    Eigen::Vector3d eta1(dist(gen), dist(gen), dist(gen));
    Eigen::Vector3d eta2(dist(gen), dist(gen), dist(gen));
    
    walker.pos1 += eta1;
    walker.pos2 += eta2;
}

bool testAcceptance(Walker walkerOld, Walker walker, double alpha, std::mt19937& gen)
{
    // Calculate transition probability from walker --> walkerOld
    double r12New = (walker.pos1 - walker.pos2).norm();
    double prefactor = 1/(std::sqrt(2*M_PI*dtau));
    Eigen::Vector3d F1New = -4*walker.pos1.normalized() + (walker.pos1 - walker.pos2) / (r12New * (1 + alpha * r12New) * (1 + alpha * r12New));
    Eigen::Vector3d F2New = -4*walker.pos2.normalized() + (walker.pos2 - walker.pos1) / (r12New * (1 + alpha * r12New) * (1 + alpha * r12New));
    double D1_newToOld = (walkerOld.pos1 - walker.pos1 - (dtau/2)*F1New).squaredNorm();
    double D2_newToOld = (walkerOld.pos2 - walker.pos2 - (dtau/2)*F2New).squaredNorm();
    double T_newToOld = prefactor*std::exp(-(D1_newToOld + D2_newToOld)/(2*dtau));

    // Calculate transition probability from walkerOld --> walker
    double r12Old = (walkerOld.pos1 - walkerOld.pos2).norm();
    Eigen::Vector3d F1Old = -4*walkerOld.pos1.normalized() + (walkerOld.pos1 - walkerOld.pos2) / (r12Old * (1 + alpha * r12Old) * (1 + alpha * r12Old));
    Eigen::Vector3d F2Old = -4*walkerOld.pos2.normalized() + (walkerOld.pos2 - walkerOld.pos1) / (r12Old * (1 + alpha * r12Old) * (1 + alpha * r12Old));
    double D1_oldToNew = (walker.pos1 - walkerOld.pos1 - (dtau/2)*F1Old).squaredNorm();
    double D2_oldToNew = (walker.pos2 - walkerOld.pos2 - (dtau/2)*F2Old).squaredNorm();
    double T_oldToNew = prefactor*std::exp(-(D1_oldToNew + D2_oldToNew)/(2*dtau));

    // Calulate value of guide function at walker position
    double psiTnew = trialWavefunctionSquared(walker, alpha);

    // Calculate value of guide function at walkerOld position
    double psiTold = trialWavefunctionSquared(walkerOld, alpha);
    
    // Calculate A = T_newToOld * psiTnew / T_oldToNew * psiTold
    double A = (T_newToOld * psiTnew) / (T_oldToNew * psiTold);

    // Acceptance Criteria
    if (A >= 1)
    {
        return true;
    }
    else 
    {
        std::uniform_real_distribution<double> dist(0, 1);
        double x = dist(gen);
        if (A >= x)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

int main()
{   
    
    // Open file to store results and write headers
    std::ofstream outFile("results.dat");
    outFile << "Step, Population, Average Energy, Variance" << std::endl;

    // RNG
    std::random_device rd;
    std::mt19937 gen(rd());

    // Constants
    double gamma = std::sqrt(dtau);
    double EZero = -2.9;
    int equilibrationSteps = 10000;
    std::vector<double> alphas = {0.15};
    double learningRate = 0.01;
    double ET = EZero;

   
    // Main loop
    for (int alpha = 0; alpha < static_cast<int>(alphas.size()); alpha++)
    {
        // Initialise Walkers 
        std::vector<Walker> walkers(numWalkersTarget);
        initialiseWalkers(walkers);

        std::cout << "alpha: " << alphas[alpha] << std::endl;
        
        // Accumulators
        double sumLocalEnergy = 0.0;
        int countLocalEnergy = 0;

        for (int step = 0; step < numSteps; step++)
        {

            // Container to hold new population of walkers (i.e; those that were not deleted and the newly birthed ones)
            std::vector<Walker> newWalkers;

            for (Walker& walker : walkers)
            {   
                // Record current position for acceptance test
                Walker walkerOld = walker;

                // Shift walker according to Fokker-Planck force
                shiftFokkerPlanck(walker, alphas[alpha]);
                
                // Shift walker accoring to η gaussian distribution
                shiftEtaGaussian(walker, gamma, gen);

                // Test acceptance
                bool isAccepted = testAcceptance(walkerOld, walker, alphas[alpha], gen);
                
                // If move rejected put walker back
                if (isAccepted == false)
                {
                    walker = walkerOld;
                    newWalkers.push_back(walker);
                    
                    // Discard equilibration steps
                    if (step > equilibrationSteps)
                    {   
                        // Accumulate energy
                        double energy = localEnergy(walker, alphas[alpha]);
                        sumLocalEnergy += energy;
                        countLocalEnergy++;
                    }

                }

                // If move accepted evaluate q and delete or birth walker
                else if (isAccepted == true)
                {   
                   double q = std::exp(-dtau*(localEnergy(walker, alphas[alpha]) + localEnergy(walkerOld, alphas[alpha]))/2 + dtau*ET);

                   std::uniform_real_distribution<double> dist(0, 1);
                   double r = dist(gen);
                   double s = q + r;
                   int sInteger = static_cast<int>(std::floor(s)); // Truncate s to obtain only integer part
                    
                   // If sInteger == 0 --> Do nothing --> Walker is not carried forward into next step and hence deleted

                   // Birth integer part of s walkers at old position for s > 0
                   if (sInteger > 0)
                   {
                        for (int i = 0; i < sInteger; i++)
                        {
                            newWalkers.push_back(walker);
                            
                            // Discard equilibration steps
                            if (step > equilibrationSteps)
                            {   
                                // Accumulate energy
                                double energy = localEnergy(walker, alphas[alpha]);
                                sumLocalEnergy += energy;
                                countLocalEnergy++;
                            }
                        }
                   }
                }
            }
            
            // Carry forward surviving walkers
            walkers = std::move(newWalkers);

            // Check population is not empty
            if (walkers.empty()) 
            {
                std::cerr << "All walkers have died at step " << step << ". Terminating simulation.\n";
                break;
            }

            // Update ET
            ET += learningRate * std::log(static_cast<double>(numWalkersTarget) / walkers.size());
 
            // Output progress
            double localEnergyCurrent = sumLocalEnergy/countLocalEnergy;

            if (step % 1000 == 0)
            {    
                if (std::isnan(localEnergyCurrent))
                {
                    std::cout << "Step: " << step << ", Population: " << walkers.size() << ", Average energy: " << "Still equilibrating" << std::endl;
                }
                else
                {
                    std::cout << "Step: " << step << ", Population: " << walkers.size() << ", Average energy: " 
                              << localEnergyCurrent << std::endl;

                    outFile << step << ", " << walkers.size() << ", " << localEnergyCurrent << std::endl; // Write to file
                }
            }

        }
        
        // Caluclate final energy
        double energy = sumLocalEnergy/countLocalEnergy;
        std::cout << "Ground State Energy: " << energy << " for alpha: " << alphas[alpha] << std::endl;
        std::cout << std::endl;
    }
    return 0;
}
