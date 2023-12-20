import numpy as np
import sympy

E = sympy.Symbol('E')

m_e = 1

hbar = 1

#V = E_c - 2*(p**2)*(m_c/(m_e**2))



#%%

def effective_masses(E_clist, m_clist, plist): #function calculating the effective mass of each layer
    mlist = []
    for i in range(0, len(m_clist))
        V = E_clist[i] - 2*(plist[i]**2)*({m_clist[i]}/(m_e**2)) #valence energy band
        m = ((m_e**2)*(E - V)/(2*(plist[i]**2)))  #expression for effective mass
        mlist.append(m)  #creates list of effective masses as a function of energy
        
#%%

def lambdda(mlist, E_clist):
    l_list = []
    for i range(0, len(mlist)):
        l = (np.sqrt(2*mlist[i]*E*(E - {E_clist[i]})))/hbar#expression for lambda, dependent on energy


#%%

def alpha(mlist, l_list):
    alist = []
    for i range(0, len(mlist)):
        a = (mlist[i-1]*l_list[i])/(mlist[i]*l_list[i-1])
                

#%%
thlist = [3.1, 7.1, 2.1, 14.2]
mcn = [0.0919, 0.067, 0.0919, 0.067] 
Ecn = [0.303, 0, 0.303, 0]
        
for i in range(len(Ecn)):
    print(Ecn[i], mcn[i], Ecn[i-1], mcn[i-1], thlist[i])
    
#%%

# Define the initial bounds of the energy search interval
E_min_initial = 0.015
E_max_initial = 0.025

# Set the tolerance level
tolerance = 1e-10

# Initialize a list to store the roots
roots = []

# Run the bisection method 5 times
for i in range(10):
    # Reset the search interval to the initial values
    E_min = E_min_initial
    E_max = E_max_initial

    # Apply the bisection method
    while (E_max - E_min) > tolerance:
        E_mid = (E_min + E_max) / 2
        if np.sign(det_func(E_min)) != np.sign(det_func(E_mid)):
            E_max = E_mid
        else:
            E_min = E_max +  0.01
            
    if 0 < np.real(det_func(E_mid)) <= 1e-2:
        # Add the root to the list
        roots.append(E_mid)
        
    E_min_initial = E_mid + 0.01
    E_max_initial = E_min_initial + 0.01

# Print the list of roots
print("Roots:", roots)
#%%

E = 0.38
C = np.sqrt(9.11e-31*1.6e-19)/ 1.05e-34
Ec1 = 0.303
Ec2 = 0
Ev1 = -1.780373
Ev2 = -1.5188900000000003



m1 = ((E - Ev1))/(p)
m2 = ((E - Ev2))/(p)
l1 = np.sqrt(2*m1*(Ec1 - E) + 0j)*C
l2 = np.sqrt(2*m2*(Ec2 - E) + 0j)*C

a = (m2*l1)/(m1*l2) 

print(np.real(a), np.real(1/a))

#%%

def rootsdf(Nq):
    roots_list = []

    for i in range(1, Nq + 1):
        q = -np.pi/d - np.pi/(Nq*d) + 2*np.pi*i/(Nq*d)
        roots = rootfinder(q)
        roots_dict = {f"Root {j}": root for j, root in enumerate(roots, start=1)}
        roots_dict["q"] = q*d
        roots_list.append(roots_dict)

    df = pd.DataFrame(roots_list)

    # Move the q column to the first position
    cols = df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    # save the dataframe to a csv file
    #df.to_csv("roots2.csv", index=False)
    
    return df[cols]

 def bandsplot(Nq):
     bands = rootsdf(Nq)
     # Define the x and y data
     x_data = bands['q']
     y_data = bands.iloc[:, 1:]  # select all columns except the first (q)

     # Plot each column with a label
     for col in y_data.columns:
         plt.plot(x_data, y_data[col], label=col)
         #plt.scatter(x_data, y_data[col], label=col)

     # Add labels and legend
     plt.xlabel('q (1/d)')
     plt.ylabel('Energy (eV)')
     #plt.legend()
     
     #save plot
     #plt.savefig('roots_plot.png')

     # Show the plot
     return plt.show()
 

def get_layer_index(z):# for a given length, find which layer we are in.
    if z < sum(thlist):
        z = z
    else:
        z = z % sum(thlist)
    for i in range(1, len(thlist) + 1):
        if sum(thlist[:i-1]) < z < sum(thlist[:i]):
            return i
    

def layerprods(E, z):#calculate the matrix products for each layer, from the first layer
    mx = layer_matrices(E)
    n = get_layer_index(z)
    result = mx[0]
    for i in mx[1:n]:
        result = np.matmul(i, result)
    return result
           



