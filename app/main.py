from tkinter import *
from tkinter.ttk import *
import subprocess
from process_data import ProcessData

class Test():
    def __init__(self):

        self.processData = ProcessData()

        # print(self.processData )
        self.root = Tk()
        self.root.geometry('300x250')
        self.button = Button(self.root,
                          text = 'Click Me',
                          command=lambda:self.pop_up())

        self.button.pack()

          

        self.pb1 = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='indeterminate')
        self.pb1.pack(expand=True)

        self.root.mainloop()            
        self.text_entry = str()




    def set_e(self, text):

        self.text_entry = text
        self.root.update_idletasks()
        self.pb1['value'] += 20
        # print(text)
    def scrape(self):
        # pass
        # print(self.e.get())
        process = subprocess.Popen(f"scrapy runspider Scrape_AmazonReviews/Scrape_AmazonReviews/spiders/AmazonReviews.py -o output.csv -a myBaseUrl={self.text_entry}")
        process.wait()
        true_lie_dir, len_data = self.processData.process_data()

        self.lb1 = Label(self.root, 
                      text = f"Trust: {str(true_lie_dir[1])}").place(x = 40,
                                               y = 100)  
        self.lb2 = Label(self.root, 
                      text = f"Fake: {str(true_lie_dir[0])}").place(x = 40,
                                               y = 120)

        self.lb3 = Label(self.root, 
                      text = f"Total: {len_data}").place(x = 40,
                                               y = 150)

        pass

    def pop_up(self):
        top = Toplevel()
        btn = Button(top, text="URL here: ")
        e = Entry(top, width = 47     )
        print("e.get()", e)
        # self.text_entry = 
        scrape = Button(top, text="Start", command=lambda: [self.set_e(e.get()), top.destroy(), self.scrape()])

        # btn.grid(row=0, column=0)
        btn.place(x = 5, y = 10)           
        # e.grid(row=1, column=100)
        e.place(x = 5, y = 40)
        scrape.place(x = 5, y = 68)
        # scrape.grid(row=2, column=0)


app = Test()