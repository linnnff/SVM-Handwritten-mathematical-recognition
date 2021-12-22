import wx
from collections import namedtuple
from PIL import Image
import os
from model_f import Tester
 
origin_path = os.getcwd()
wildcard ="png (*.png)|*.png|" \
           "jpg(*.jpg) |*.jpg|"\
           "jpeg(*.jpeg) |*.jpeg|"\
           "tiff(*.tif) |*.tiff|"\
           "All files (*.*)|*.*"
class MainWindow(wx.Frame):
    def __init__(self,parent,title):
        wx.Frame.__init__(self,parent,title=title,size=(600,-1))
        static_font = wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL)
        Size = namedtuple("Size",['x','y'])
        s = Size(100,50)
        model_path = os.path.join(origin_path,'mnist_svm.m')
        self.fileName = None
        self.model = Tester(model_path)
        b_labels = [u'open',u'run']
        TipString = [u'选择图片', u'识别数字']
        funcs = [self.choose_file,self.run]
        '''create input area'''
        self.in1 = wx.TextCtrl(self,-1,size = (2*s.x,3*s.y))
        self.out1 = wx.TextCtrl(self,-1,size = (s.x,3*s.y))
        '''create button'''
        self.sizer0 = wx.FlexGridSizer(cols=4, hgap=4, vgap=2) 
        self.sizer0.Add(self.in1)
        buttons = []
        for i,label in enumerate(b_labels):
            b = wx.Button(self, id = i,label = label,size = (1.5*s.x,s.y))
            buttons.append(b)
            self.sizer0.Add(b)      
        self.sizer0.Add(self.out1)
        '''set the color and size of labels and buttons'''  
        for i,button in enumerate(buttons):
            button.SetForegroundColour('red')
            button.SetFont(static_font)
#            button.SetToolTipString(TipString[i]) #wx2.8
            button.SetToolTip(TipString[i])   #wx4.0
            button.Bind(wx.EVT_BUTTON,funcs[i])
        '''layout'''
        self.SetSizer(self.sizer0)
        self.SetAutoLayout(1)
        self.sizer0.Fit(self)
        self.CreateStatusBar()
        self.Show(True)
    def run(self,evt):
        if self.fileName is None:
            self.raise_msg(u'请选择一幅图片')
            return None
        else:
            ans = self.model.predict(self.fileName)
            self.out1.Clear()
            self.out1.write(str(ans))
    def choose_file(self,evt):
        '''choose img'''
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=os.getcwd(), 
            defaultFile="",
            wildcard=wildcard,
#            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR #wx2.8
            style = wx.FD_OPEN | wx.FD_MULTIPLE |     #wx4.0
                    wx.FD_CHANGE_DIR | wx.FD_FILE_MUST_EXIST |
                    wx.FD_PREVIEW
            )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            dlg.Destroy()
            self.in1.Clear()
            self.in1.write(paths[0])
            self.fileName = paths[0]
            im = Image.open(self.fileName)
            im.show()
        else:
            return None
    def raise_msg(self,msg):
        '''warning message'''
        info = wx.AboutDialogInfo()
        info.Name = "Warning Message"
        info.Copyright = msg
        wx.AboutBox(info)
if __name__ == '__main__':
    app = wx.App(False)
    frame = MainWindow(None,'Digit Recognize')
    app.MainLoop()