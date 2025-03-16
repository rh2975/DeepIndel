#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int main()
{

    ifstream infile("del.vcf");
    ofstream outfile("my_deletions.vcf");

    int a, b;
    string ss, dd, kk;

    while (getline(infile, ss))
    {
        

        if (ss.substr(0, 3) == "chr")
        {
            //cout<<ss<<endl;
            istringstream iss(ss);
            vector<string> tokens;
            string token;
            while (getline(iss, token, '\t') || getline(iss, token, ' ')) // but we can specify a different one
                tokens.push_back(token);
            
  
            for (int i = 0; i <= 1; i++)
            {
                outfile <<tokens[i]<<'\t';
            }
            outfile<<"."<<'\t'<<tokens[3]<<'\t';	
            for (int i = 4; i < tokens.size()+2; i++)
            {
                outfile <<"."<<'\t';
            }

            outfile<<"."<<endl;
        }
        else outfile<<ss<<endl;
    }
    outfile.close();
    return 0;
}
