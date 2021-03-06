

# Introduction to Linux

## Preparation

1. Boot from a usb stick (or live cd), we suggest to use  [Ubuntu gnome](http://ubuntugnome.org/) distribution, or another ubuntu derivative.

2. (Optional) Configure keyboard layout and software repository
   Go to the the *Activities* menu (top left corner, or *start* key):
      -  Go to settings, then keyboard. Set the layout for latin america
      -  Go to software and updates, and select the server for Colombia
3. (Optional) Instead of booting from a live Cd. Create a partition in your pc's hard drive and install the linux distribution of your choice, the installed Os should perform better than the live cd.

## Introduction to Linux

1. Linux Distributions

   Linux is free software, it allows to do all sort of things with it. The main component in linux is the kernel, which is the part of the operating system that interfaces with the hardware. Applications run on top of it. 
   Distributions pack together the kernel with several applications in order to provide a complete operating system. There are hundreds of linux distributions available. In
   this lab we will be using Ubuntu as it is one of the largest, better supported, and user friendly distributions.


2. The graphical interface

   Most linux distributions include a graphical interface. There are several of these available for any taste.
   (http://www.howtogeek.com/163154/linux-users-have-a-choice-8-linux-desktop-environments/).
   Most activities can be accomplished from the interface, but the terminal is where the real power lies.

### Playing around with the file system and the terminal
The file system through the terminal
   Like any other component of the Os, the file system can be accessed from the command line. Here are some basic commands to navigate through the file system

   -  ``ls``: List contents of current directory
   - ``pwd``: Get the path  of current directory
   - ``cd``: Change Directory
   - ``cat``: Print contents of a file (also useful to concatenate files)
   - ``mv``: Move a file
   - ``cp``: Copy a file
   - ``rm``: Remove a file
   - ``touch``: Create a file, or update its timestamp
   - ``echo``: Print something to standard output
   - ``nano``: Handy command line file editor
   - ``find``: Find files and perform actions on it
   - ``which``: Find the location of a binary
   - ``wget``: Download a resource (identified by its url) from internet 

Some special directories are:
   - ``.`` (dot) : The current directory
   -  ``..`` (two dots) : The parent of the current directory
   -  ``/`` (slash): The root of the file system
   -  ``~`` (tilde) :  Home directory
      
Using these commands, take some time to explore the ubuntu filesystem, get to know the location of your user directory, and its default contents. 
   
To get more information about a command call it with the ``--help`` flag, or call ``man <command>`` for a more detailed description of it, for example ``man find`` or just search in google.


## Input/Output Redirections
Programs can work together in the linux environment, we just have to properly 'link' their outputs and their expected inputs. Here are some simple examples:

1. Find the ```passwd```file, and redirect its contents error log to the 'Black Hole'
   >  ``find / -name passwd  2> /dev/null``

   The `` 2>`` operator redirects the error output to ``/dev/null``. This is a special file that acts as a sink, anything sent to it will disappear. Other useful I/O redirection operations are
      -  `` > `` : Redirect standard output to a file
      -  `` | `` : Redirect standard output to standard input of another program
      -  `` 2> ``: Redirect error output to a file
      -  `` < `` : Send contents of a file to standard input
      -  `` 2>&1``: Send error output to the same place as standard output

2. To modify the content display of a file we can use the following command. It sends the content of the file to the ``tr`` command, which can be configured to format columns to tabs.

   ```bash
   cat milonga.txt | tr '\n' ' '
   ```
   
## SSH - Server Connection

1. The ssh command lets us connect to a remote machine identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER (**vision** in our case). The second command allows us to copy files between systems (you will get the actual login information in class).

   ```bash
   
   #connect
   ssh USER@SERVER
   ```

2. The scp command allows us to copy files form a remote server identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER. Following the SERVER information, we add ':' and write the full path of the file we want to copy, finally we add the local path where the file will be copied (remember '.' is the current directory). If we want to copy a directory we add the -r option. for example:

   ```bash
   #copy 
   scp USER@SERVER:~/data/sipi_images .
   
   scp -r USER@SERVER:/data/sipi_images .
   ```
   
   Notice how the first command will fail without the -r option

See [here](ssh.md) for different types of SSH connection with respect to your OS.

## File Ownership and permissions   

   Use ``ls -l`` to see a detailed list of files, this includes permissions and ownership
   Permissions are displayed as 9 letters, for example the following line means that the directory (we know it is a directory because of the first *d*) *images*
   belongs to user *vision* and group *vision*. Its owner can read (r), write (w) and access it (x), users in the group can only read and access the directory, while other users can't do anything. For files the x means execute. 
   ```bash
   drwxr-x--- 2 vision vision 4096 ene 25 18:45 images
   ```
   
   -  ``chmod`` change access permissions of a file (you must have write access)
   -  ``chown`` change the owner of a file
   
## Sample Exercise: Image database

1. Create a folder with your Uniandes username. (If you don't have Linux in your personal computer)

2. Copy *sipi_images* folder to your personal folder. (If you don't have Linux in your personal computer)

3.  Decompress the images (use ``tar``, check the man) inside *sipi_images* folder. 

4.  Use  ``imagemagick`` to find all *grayscale* images. We first need to install the *imagemagick* package by typing

    ```bash
    sudo apt-get install imagemagick
    ```
    
    Sudo is a special command that lets us perform the next command as the system administrator
    (super user). In general it is not recommended to work as a super user, it should only be used 
    when it is necessary. This provides additional protection for the system.
    
    ```bash
    find . -name "*.tiff" -exec identify {} \; | grep -i gray | wc -l
    ```
    
3.  Create a script to copy all *color* images to a different folder
    Lines that start with # are comments
       
      ```bash
      #!/bin/bash
      
      # go to Home directory
      cd ~ # or just cd

      # remove the folder created by a previous run from the script
      rm -rf color_images

      # create output directory
      mkdir color_images

      # find all files whose name end in .tif
      images=$(find sipi_images -name *.tiff)
      
      #iterate over them
      for im in ${images[*]}
      do
         # check if the output from identify contains the word "gray"
         identify $im | grep -q -i gray
         
         # $? gives the exit code of the last command, in this case grep, it will be zero if a match was found
         if [ $? -eq 0 ]
         then
            echo $im is gray
         else
            echo $im is color
            cp $im color_images
         fi
      done
      
      ```
      -  save it for example as ``find_color_images.sh``
      -  make executable ``chmod u+x`` (This means add Execute permission for the user)
      -  run ``./find_duplicates.sh`` (The dot is necessary to run a program in the current directory)
      

## Your turn

1. What is the ``grep``command?

The grep command is used to search for patterns in files. In order to get this
information, the command "man grep" was used on the terminal.

2. What is the meaning of ``#!/bin/python`` at the start of scripts?

The #!/usr/bin/python line at the beginning of a code makes it easier to execute python codes,
because it is no longer necessary to specify "Python" before executing, it's enough to type ./filename.py to execute it, as long as the file has the needed permission.

![cositoraro](https://user-images.githubusercontent.com/47038625/52246504-1ac73d00-28b4-11e9-9e64-aa1f37648222.png)

This python program tells the user wether the entered number is prime or not, it can be clearly seen that it can be used by specifying python to run it or using the aforementioned syntaxis.

3. Download using ``wget`` the [*bsds500*](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) image segmentation database, and decompress it using ``tar`` (keep it in you hard drive, we will come back over this data in a few weeks).

The database was downloaded by copying the download link into the terminal and using the "wget -r" to download it into the selected folder.

![screenshot from 2019-01-31 18-05-31](https://user-images.githubusercontent.com/47038625/52248684-9679b780-28bd-11e9-90b7-d763b5b6f88a.png)

After downloading the compressed file, the following command was used in order to uncompress it:  "tar -xvzf BSR_bsds500.tgz". The result was satisfactory. 
 
4. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?

The command "du -h -c" was used in order to get this information. "du" returns the disk usage of the file inside the given folder, the "-h" flag converts the result returned by the original function into "human" readable form (b, kb, mb, etc.) and the "-c" flag returns the total disk usage of all files in the selected folder. The results were the following:

![du](https://user-images.githubusercontent.com/47038625/52251392-916f3500-28ca-11e9-8c6f-69c05236add1.png)

in which we can read that the total disk size of the uncompressed file is of 73 mb.

In order to get the amount of images in the requested folder, the following code was written:

![count](https://user-images.githubusercontent.com/47038625/52249507-7cda6f00-28c1-11e9-85aa-a18b48a91f01.png)

This code counts the amount of files that end with .jpg. knowing that all images in this folders use this format (which was made sure using the "ls" command on all three folders), this code is counting the amount of images in them. There's a total of 500 images in this folder. The code shown also creates a .text file containing information for every image in the folder. this will be used later. 
 
5. What are all the different resolutions? What is their format? Tip: use ``awk``, ``sort``, ``uniq``

Using the .text file created in the last exercise, the next code was made: 

![all_dimension](https://user-images.githubusercontent.com/47038625/52249832-0fc7d900-28c3-11e9-9c2a-23992aba28d6.png)

In this code, the command "awk" is used, in order to only get the dimension informaton of the images. This is written in another .text file.
After that, the command "sort" is used so all dimensions are ordered. This must be done so the command "uniq" works, as it only deletes equal information when it's next to each other. Doing this, we can conlude the only dimensions present in this folder are:
321x481 & 481x321, which represent portrait and landscape orientation respectively.

6. How many of them are in *landscape* orientation (opposed to *portrait*)? Tip: use ``awk`` and ``cut``

To get the amount of images in Portrait and Landscape orientation, the following code was made:

![port_land](https://user-images.githubusercontent.com/47038625/52250176-8e714600-28c4-11e9-8a8b-4cd970bcb219.png)

This code transposes the .text file containing the dimensions of all images, then using the command "cut" and other manipulations, compares it to "481x321", being this the only Landscape orientation, the code sums 1 to the landscape pictures counter, otherwise sums 1 to the portrait pictures one. This processes is iterated over all images inside the folder in question. After that, prints both numbers on the terminal.
The amount of Portrait oriented pictures is 152, while the amount of Landscape oriented pictures is 348.
 
7. Crop all images to make them square (256x256) and save them in a different folder. Tip: do not forget about  [imagemagick](http://www.imagemagick.org/script/index.php).

In order to crop all images to square size and put them in a new folder the following code was written:

![cropping](https://user-images.githubusercontent.com/47038625/52250530-50752180-28c6-11e9-88f4-ebc043b4cbe5.png)

This code creates a new folder called cropped_images and then, again, iterates over all images in the last folder. the processing consists of using the command "convert" with the flag "-crop" and giving the parameter 256x256+0+0, which simply says that the cropping will go from the superior left corner 256 pixel down and 256 right. The name given to each crop will be a number from 1 to the total amount of images, which is done using a counter and using it as a name.
The resulting folder looks like this:

![screenshot from 2019-02-04 22-01-43](https://user-images.githubusercontent.com/47038625/52250979-84e9dd00-28c8-11e9-98e2-01292bd6a2fe.png) 


Complete Code:

```bash
cd ~
cd /home/santiago/Documents/BSDS300/BSR/BSDS500/data
 ##Folder where val, train & test images are

rm -rf images_info.txt
# find all files whose name end in .jpg
images_=$(find images -name *.jpg)
#iterate over them
COUNTER=0
for im in ${images_[*]}
do
  identify $im>>images_info.txt
  COUNTER=$[$COUNTER +1]
   done; 
echo "Number of images:"
echo ${COUNTER[*]}
###USE OF THE .TXT FILE
rm -rf images_dimensions.txt

awk '{print $3}' images_info.txt>images_dimensions.txt
echo "Unique Dimension:"
sort images_dimensions.txt | uniq 
### 
rm -rf tr_img_dim.txt
tr '\n' ',' < images_dimensions.txt > tr_img_dim.txt

INIT=1
ENDD=7

PORTRAIT=0
LADNSCAPE=0

for im in ${images_[*]}
do	
aux=$(cut -c ${INIT[*]}-${ENDD[*]}  tr_img_dim.txt)
INIT=$[$INIT +8]
ENDD=$[$ENDD +8]

   if [ ${aux} = '481x321' ]
   then
      LANDSCAPE=$[LANDSCAPE+1]
   else
      PORTRAIT=$[PORTRAIT+1]
   fi

done;
echo Number of Portrait vs Landscape images:
echo Portrait=$PORTRAIT
echo Landscape=$LANDSCAPE
### Cropping
rm -rf cropped_images
mkdir cropped_images
COUNTER=1

for im in ${images_[*]}
do
convert $im -crop 256x256+0+0 $COUNTER.jpg

mv $COUNTER.jpg /home/santiago/Documents/BSDS300/BSR/BSDS500/data/cropped_images

COUNTER=$[$COUNTER +1]
done;
echo Folder finished


rm -rf images_info.txt
rm -rf images_dimensions.txt
rm -rf tr_img_dim.txt
```


# References:

https://askubuntu.com/questions/25347/what-command-do-i-need-to-unzip-extract-a-tar-gz-file
Extract .tar

https://www.lifewire.com/write-awk-commands-and-scripts-2200573
use awk

https://unix.stackexchange.com/questions/76049/what-is-the-difference-between-sort-u-and-sort-uniq
use sort & uniq

https://www.computerhope.com/unix/ucut.htm
use cut

https://www.tecmint.com/check-linux-disk-usage-of-files-and-directories/
get disk usage 

https://stackoverflow.com/questions/10515964/counter-increment-in-bash-loop-not-working
make a counter

https://odin.mdacc.tmc.edu/~ryu/linux.html
transpose a .text file

http://www.imagemagick.org/Usage/crop/
crop images

# Report

For every question write a detailed description of all the commands/scripts you used to complete them. DO NOT use a graphical interface to complete any of the tasks. Use screenshots to support your findings if you want to. 

Feel free to search for help on the internet, but ALWAYS report any external source you used.

Notice some of the questions actually require you to connect to the course server, the login instructions and credentials will be provided on the first session. 

## Deadline

We will be delivering every lab through the [github](https://github.com) tool (Silly link isn't it?). According to our schedule we will complete that tutorial on the second week, therefore the deadline for this lab will be specially long **February 7 11:59 pm, (it is the same as the second lab)** 

### More information on

http://www.ee.surrey.ac.uk/Teaching/Unix/ 




